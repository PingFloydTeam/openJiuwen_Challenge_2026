"""Contest entry point."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import time

from ping_floyd.domain import AnswerLog, Task
from ping_floyd.planning import Planner
from ping_floyd.proxy import Budget, MODELS, Proxy
from ping_floyd.tool_menu import TOOL_PROFILES


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=os.getenv("TASKS") or None)
    parser.add_argument("--out", type=Path, default=os.getenv("OUT") or None,
                        help="optional plain text snapshot path or writable directory")
    parser.add_argument("--budget", type=int, default=os.getenv("BUDGET", "100000"))
    parser.add_argument("--task-cap", type=int, default=3200)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repair-limit", type=int, default=3, help="maximum code edits per model pass")
    parser.add_argument("--queue", choices=("value", "reward"), default="value")
    parser.add_argument("--calibration", type=Path, help="cost and success priors learned on development tasks")
    parser.add_argument("--fixed-estimates", action="store_true", help="disable cost updates during the run")
    parser.add_argument("--retry-share", type=int, default=10, help="maximum percent spent on a second model")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=1,
                        help="concurrent model workflows, after the local pass")
    parser.add_argument("--seconds", type=int, default=3500)
    parser.add_argument("--model", choices=sorted(MODELS),
                        default=os.getenv("LLM_MODEL", "qwen3.8-27b-noreason"))
    parser.add_argument("--retry-model", choices=sorted(MODELS),
                        help="optional second model for stalled tasks, after first attempts")
    parser.add_argument("--tool-profile", choices=TOOL_PROFILES,
                        default=os.getenv("LLM_TOOL_PROFILE", "compact"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--agent-only", action="store_true", help="disable deterministic solvers for model comparison")
    mode.add_argument("--offline", action="store_true", help="use only deterministic solvers and checks")
    parser.add_argument("--reasoning-separate", action="store_true",
                        help="use only if proxy explicitly excludes reasoning from completion_tokens")
    parser.add_argument("--trace", type=Path, help="optional JSONL event log; use a writable directory")
    args = parser.parse_args()
    if args.model not in MODELS or min(args.budget, args.task_cap, args.steps, args.seconds, args.repair_limit) <= 0:
        parser.error("use positive limits and an allowed model")
    if not 0 <= args.retry_share <= 25:
        parser.error("retry share must be between 0 and 25 percent")
    if args.tool_profile not in TOOL_PROFILES:
        parser.error("use an allowed tool profile")
    if args.tasks_dir is None or not args.tasks_dir.is_dir():
        parser.error("tasks directory does not exist")
    if args.agent_only and not os.getenv("LLM_API_KEY"):
        parser.error("--agent-only requires LLM_API_KEY")
    if args.retry_model and (args.retry_model == args.model or args.offline):
        parser.error("--retry-model requires a different model and an online run")
    return args


async def main(args):
    from openjiuwen.core.common.logging.log_config import configure_log_config

    configure_log_config({"level": "WARNING", "output": ["console"], "log_path": "/tmp/jiuwen-logs"})
    from openjiuwen.core.runner.runner import Runner
    from ping_floyd.agent import run_task

    def trace(event):
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        print(line, flush=True)
        if args.trace:
            with args.trace.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    tasks = []
    for manifest in sorted(args.tasks_dir.resolve().glob("*/task.json")):
        if manifest.is_symlink() or manifest.parent.is_symlink():
            continue
        try:
            tasks.append(Task.load(manifest))
        except (OSError, ValueError, KeyError, TypeError):
            trace({"event": "invalid_manifest", "task": manifest.parent.name})
    if not tasks:
        raise SystemExit("no task manifests found")
    try:
        answer_log = AnswerLog(args.out, tasks, trace) if args.out else None
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid OUT: {error}") from None
    if answer_log and answer_log.dirty:
        raise SystemExit("OUT could not be initialized; use a writable mounted directory")
    budget = Budget(args.budget * 10)
    deadline = time.monotonic() + args.seconds
    key = os.getenv("LLM_API_KEY")
    planner = Planner(args.task_cap * 10, args.queue, not args.fixed_estimates)
    if args.calibration:
        try:
            planner.load(args.calibration)
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as error:
            raise SystemExit(f"invalid planner calibration: {error}") from None
    proxy = Proxy(os.getenv("LLM_BASE_URL", "https://api.neuraldeep.ru/v1"),
                  key, args.model, budget, trace, deadline, args.reasoning_separate)
    submitted = 0
    submitted_reward = 0
    pending = []
    retry = []
    attempted = 0
    retried = 0
    active = {}
    unvisited = len(tasks)
    cancelled = 0
    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    proxy.stopping = stopping
    stop_reason = None

    def request_stop(reason):
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
            stopping.set()
            trace({"event": "stop_requested", "reason": reason})

    def saved_answer(task, answer):
        if answer_log:
            answer_log.record(task, answer)
            if answer_log.dirty:
                request_stop("out_write_error")

    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_signals:
        loop.add_signal_handler(sig, request_stop, sig.name)
    deadline_timer = loop.call_later(max(0, deadline - time.monotonic()), request_stop, "deadline")
    stop_waiter = asyncio.create_task(stopping.wait())

    async def evaluate(task, offline, cap):
        task_id = task.root.name
        before = budget.paid_by_task.get(task_id, 0)
        started = time.monotonic()
        try:
            result = await run_task(task, proxy, trace, cap, args.steps, not offline, offline,
                                    args.tool_profile, args.repair_limit, saved_answer if answer_log else None)
            saved, route = result["saved"], result["route"]
        except Exception as error:
            saved, route = False, f"workflow_error:{type(error).__name__}"
        cost = budget.paid_by_task.get(task_id, 0) - before
        return saved, route, cost, round(time.monotonic() - started, 3)

    def report(task, result, phase):
        nonlocal submitted, submitted_reward
        saved, route, cost, elapsed = result
        if saved:
            submitted += 1
            submitted_reward += task.reward
        trace({"event": "task", "task": task.root.name, "kind": task.kind, "reward": task.reward,
               "saved": saved, "route": route, "phase": phase, "cost_tenths": cost,
               "elapsed_s": elapsed})

    async def completed(futures):
        nonlocal cancelled
        done, _ = await asyncio.wait({*futures, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
        if stop_waiter not in done:
            return done
        finished, incomplete = await asyncio.wait(futures, timeout=2)
        for future in incomplete:
            future.cancel()
        await asyncio.gather(*incomplete, return_exceptions=True)
        for future in incomplete:
            trace({"event": "task_cancelled", "task": active.pop(future).root.name})
        cancelled += len(incomplete)
        return finished

    trace({"event": "run_start", "tasks": len(tasks), "model": args.model,
           "budget_tenths": budget.limit, "agent_only": args.agent_only, "offline": args.offline,
           "workers": args.workers, "tool_profile": args.tool_profile, "queue": args.queue,
           "fixed_estimates": args.fixed_estimates, "calibrated": bool(args.calibration),
           "repair_limit": args.repair_limit, "retry_share": args.retry_share})
    try:
        await Runner.start()
        tasks.sort(key=lambda task: (task.kind == "coding", -task.reward, task.root.name))
        for task in tasks:
            if stopping.is_set() or time.monotonic() >= deadline:
                break
            unvisited -= 1
            if (task.root / "answer.txt").exists():
                trace({"event": "existing_answer", "task": task.root.name})
                continue
            if args.agent_only:
                pending.append(task)
                continue
            result = await evaluate(task, True, 0)
            if result[1] == "offline_unresolved" and not args.offline:
                pending.append(task)
                trace({"event": "deferred", "task": task.root.name, "kind": task.kind})
            else:
                report(task, result, "local")
        trace({"event": "local_end", "submitted": submitted, "deferred": len(pending)})
        while pending or active:
            while (pending and len(active) < args.workers and key and not budget.blocked
                   and not stopping.is_set() and time.monotonic() < deadline and budget.remaining >= 480):
                task = min(pending, key=planner.priority)
                # A task may wait for another call's reservation to be released.
                cap = planner.limit(task, budget.remaining + budget.in_flight, len(pending))
                pending.remove(task)
                attempted += 1
                trace({"event": "fallback_start", "task": task.root.name,
                       "estimated_tenths": round(10 * planner.estimate(task)), "cap_tenths": cap,
                       "model": proxy.model})
                active[asyncio.create_task(evaluate(task, False, cap))] = task
            if not active:
                break
            done = await completed(active)
            for future in sorted(done, key=lambda item: active[item].root.name):
                task = active.pop(future)
                result = future.result()
                if not budget.unknown_by_task.get(task.root.name):
                    planner.observe(task, result[2], result[1])
                report(task, result, "model")
                if args.retry_model and result[2] and result[1] in {"no_progress", "step_limit", "repair_limit"}:
                    retry.append(task)
        if args.retry_model and not pending and key and not budget.blocked:
            proxy.model = args.retry_model
            retry.sort(key=planner.priority)
            retry_start = budget.spent + budget.reserved_unknown
            for task in retry:
                remaining_retry = budget.limit * args.retry_share // 100 - (budget.spent + budget.reserved_unknown - retry_start)
                cap = min(planner.max_task_tenths, budget.remaining, remaining_retry)
                if stopping.is_set() or budget.blocked or time.monotonic() >= deadline or cap < 480:
                    break
                retried += 1
                trace({"event": "retry_start", "task": task.root.name, "model": proxy.model,
                       "cap_tenths": cap})
                future = asyncio.create_task(evaluate(task, False, cap))
                active[future] = task
                if future in await completed(active):
                    active.pop(future)
                    report(task, future.result(), "retry")
    finally:
        if active:
            for future in active:
                future.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        await proxy.client.aclose()
        await Runner.stop()
        if answer_log:
            answer_log.flush()
        deadline_timer.cancel()
        stop_waiter.cancel()
        await asyncio.gather(stop_waiter, return_exceptions=True)
        for sig, handler in previous_signals.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)
    trace({"event": "run_end", "submitted": submitted, "submitted_reward": submitted_reward,
           "spent_tenths": budget.spent, "unknown_reserved_tenths": budget.reserved_unknown,
           "remaining_tenths": budget.remaining, "model_attempted": attempted,
           "deferred_unattempted": len(pending), "model_retried": retried,
           "unvisited": unvisited, "cancelled": cancelled, "stop_reason": stop_reason,
           "out_complete": not answer_log or not answer_log.dirty})


if __name__ == "__main__":
    asyncio.run(main(options()))
