"""OpenAI-compatible transport with a shared, integer budget in tenths."""
import asyncio
from dataclasses import dataclass, field
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import hashlib
import json
import os

import httpx

MODELS = {"qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b", "qwen3.8-27b-noreason"}


class ProxyError(Exception):
    pass


class BudgetStop(ProxyError):
    pass


class TruncatedReply(ProxyError):
    pass


class RateLimited(ProxyError):
    pass


@dataclass
class Budget:
    limit: int
    spent: int = 0
    reserved_unknown: int = 0
    in_flight: int = 0
    blocked: bool = False
    paid_by_task: dict[str, int] = field(default_factory=dict)
    unknown_by_task: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self):
        return max(0, self.limit - self.spent - self.reserved_unknown - self.in_flight)

    def record(self, task_id, cost, known):
        if known:
            self.spent += cost
            ledger = self.paid_by_task
        else:
            self.reserved_unknown += cost
            ledger = self.unknown_by_task
        ledger[task_id] = ledger.get(task_id, 0) + cost

    def task_total(self, task_id):
        return self.paid_by_task.get(task_id, 0) + self.unknown_by_task.get(task_id, 0)


def token_usage(usage: dict, separate_reasoning=False):
    prompt = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")
    if (any(type(value) is not int or value < 0 for value in (prompt, completion)) or
            (reasoning is not None and (type(reasoning) is not int or reasoning < 0))):
        raise ValueError("invalid usage")
    total = usage.get("total_tokens")
    if total is not None and (type(total) is not int or total not in {prompt+completion, prompt+completion+(reasoning or 0)}):
        raise ValueError('unexplained total_tokens; usage cannot be trusted')
    if separate_reasoning or (reasoning and total == prompt + completion + reasoning):
        if reasoning is None:
            raise ValueError("provider omitted separately billed reasoning")
        completion += reasoning
    if reasoning is not None and reasoning > completion:
        raise ValueError("inconsistent reasoning usage")
    return prompt, completion, reasoning


class Proxy:
    def __init__(self, base, key, model, budget, trace, deadline, separate_reasoning=False):
        self.url = base.rstrip("/") + "/chat/completions"
        self.key = key
        self.model = model
        self.budget = budget
        self.trace = trace
        self.deadline = deadline
        self.separate_reasoning = separate_reasoning
        self.client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
        self.budget_changed = asyncio.Event()
        self.stopping = None
        self.consecutive_no_reply = 0
        self.retry_not_before = 0.0

    def no_reply(self, event):
        self.consecutive_no_reply += 1
        if self.consecutive_no_reply >= 2:
            self.budget.blocked = True
            event["proxy_disabled"] = "consecutive_no_reply"

    async def claim(self, prompt_cap, task_remaining, max_output):
        minimum = prompt_cap + 480
        while True:
            time_left = self.deadline - time.monotonic()
            if self.budget.blocked or time_left < 2 or (self.stopping is not None and self.stopping.is_set()):
                raise BudgetStop("proxy disabled or deadline reached")
            capacity = min(task_remaining, self.budget.remaining)
            output = min(max_output, (capacity - prompt_cap) // 10)
            if output >= 48:
                reservation = prompt_cap + 10 * output
                self.budget.in_flight += reservation
                return output, reservation
            if (max_output < 48 or min(task_remaining, self.budget.remaining + self.budget.in_flight) < minimum):
                raise BudgetStop("insufficient budget for another bounded call")
            try:
                async with asyncio.timeout(time_left - 1):
                    await self.budget_changed.wait()
            except TimeoutError as error:
                raise BudgetStop("deadline while waiting for reserved tokens") from error

    async def invoke(self, task_id, messages, task_remaining, max_output=900):
        start_cost = self.budget.task_total(task_id)
        for attempt in range(3):
            delay = max(0,self.retry_not_before-time.monotonic())
            if delay:
                if time.monotonic()+delay+2 >= self.deadline:
                    raise BudgetStop('rate limit exceeds remaining time')
                if self.stopping is not None:
                    try:
                        await asyncio.wait_for(self.stopping.wait(),delay)
                        raise BudgetStop('stopped during rate limit')
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(delay)
            remaining = task_remaining-(self.budget.task_total(task_id)-start_cost)
            try:
                return await self._once(task_id,messages,remaining,max_output)
            except (TruncatedReply,RateLimited) as error:
                if attempt==2:
                    raise
                self.trace({'event':'proxy_retry','task':task_id,'reason':type(error).__name__,'attempt':attempt+1})
                if isinstance(error,TruncatedReply):
                    max_output=min(4000,max_output*2)

    async def _once(self, task_id, messages, task_remaining, max_output=900):
        if not self.key:
            raise BudgetStop("LLM_API_KEY is required for model calls")
        # Conservative bound for byte-based Qwen tokenizers plus chat-template overhead.
        prompt_cap = sum(len(item["content"].encode()) for item in messages) + 128
        max_output, reservation = await self.claim(prompt_cap, task_remaining, max_output)
        started = time.monotonic()
        payload = {"model": self.model, "messages": messages,
                   "temperature": 0, "max_tokens": max_output, "stream": False}
        event = {"event": "llm", "task": task_id, "model": self.model,
                 "max_output": max_output, "reserved_tenths": reservation}
        event['request_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        if os.getenv('PF_TRACE_CONTENT') == '1':
            event['messages']=messages
        try:
            timeout = min(35, max(1, self.deadline - started))
            async with asyncio.timeout(timeout):
                response = await self.client.post(
                    self.url, headers={"Authorization": f"Bearer {self.key}"},
                    json=payload, timeout=timeout,
                )
            if response.status_code == 200 and not response.content.strip():
                raise TimeoutError("empty proxy reply")
            if response.status_code != 200:
                self.consecutive_no_reply = 0
                event["status"] = response.status_code
                if response.status_code == 429 and response.headers.get('Retry-After') is not None:
                    raw=response.headers['Retry-After']
                    try:
                        delay=float(raw)
                    except ValueError:
                        try:
                            delay=(parsedate_to_datetime(raw)-datetime.now(timezone.utc)).total_seconds()
                        except (ValueError,TypeError,OverflowError):
                            delay=-1
                    if 0<=delay<=60 and time.monotonic()+delay+2<self.deadline:
                        self.retry_not_before=max(self.retry_not_before,time.monotonic()+delay)
                        raise RateLimited('temporary HTTP 429')
                if response.status_code in {401, 402, 403, 404, 429}:
                    self.budget.blocked = True
                raise ProxyError(f"proxy HTTP {response.status_code}")
            data = response.json()
            if not isinstance(data, dict):
                raise ProxyError("malformed model response")
        except asyncio.CancelledError:
            self.budget.record(task_id, reservation, known=False)
            event.update(error="CancelledError", unknown_reserved_tenths=reservation)
            raise
        except RateLimited:
            self.budget.record(task_id,reservation,known=False)
            event.update(error='RateLimited',unknown_reserved_tenths=reservation)
            raise
        except (httpx.HTTPError, TimeoutError, KeyError, TypeError, ValueError, ProxyError) as error:
            self.budget.record(task_id, reservation, known=False)
            event.update(error=type(error).__name__, unknown_reserved_tenths=reservation)
            if isinstance(error, (httpx.TimeoutException, TimeoutError)):
                self.no_reply(event)
            raise ProxyError(str(error) if isinstance(error, ProxyError) else type(error).__name__) from None
        else:
            try:
                prompt, completion, reasoning = token_usage(data["usage"], self.separate_reasoning)
            except (AttributeError, KeyError, TypeError, ValueError):
                self.budget.record(task_id, reservation, known=False)
                event.update(usage_unavailable=True, unknown_reserved_tenths=reservation)
            else:
                cost = prompt + completion * 10
                self.budget.record(task_id, cost, known=True)
                if cost > reservation:
                    self.budget.blocked = True
                event.update(input_tokens=prompt, output_tokens=completion,
                             reasoning_tokens=reasoning, cost_tenths=cost)
            try:
                choice = data["choices"][0]
                event["finish_reason"] = choice.get("finish_reason")
                if choice.get("finish_reason") == "length":
                    self.consecutive_no_reply = 0
                    raise TruncatedReply("response truncated")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    self.no_reply(event)
                    event["error"] = "EmptyContent"
                    raise ProxyError("model returned no JSON text")
                self.consecutive_no_reply = 0
                event['response_sha256']=hashlib.sha256(content.encode()).hexdigest()
                if os.getenv('PF_TRACE_CONTENT') == '1':
                    event['response_content']=content
                    event['usage']=data.get('usage')
                return content
            except (AttributeError, KeyError, TypeError, IndexError):
                self.no_reply(event)
                event["error"] = "MalformedContent"
                raise ProxyError("malformed model response") from None
        finally:
            self.budget.in_flight -= reservation
            self.budget_changed.set()
            self.budget_changed = asyncio.Event()
            event.update(elapsed_s=round(time.monotonic() - started, 3),
                         remaining_tenths=self.budget.remaining,
                         in_flight_tenths=self.budget.in_flight)
            self.trace(event)
