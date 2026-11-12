"""Paired question forms and a live translation cost measurement."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

from .hard import Suite
from .large import dump, markdown


def generate(root, seed):
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    suites = {variant: Suite(root / variant, seed, None) for variant in ("english", "russian", "noisy")}
    missing = ("missing_information", "the requested value is underdetermined")

    def add(kind, family, index, en, ru, expected, assets=None, reason=None, check="exact_int"):
        noise = "\nЧерновик для проверки.\n\n## Условие:\n\n" + "\n\n".join(
            "• " + part if part[0].isalpha() else part for part in ru.splitlines()) + "\n\nСпасибо!\n"
        for variant, question in zip(suites, (en, ru.replace("\n", " "), noise), strict=True):
            suites[variant].add(kind, family, index, question, expected, assets=assets, reason=reason, check=check)

    for index in range(3):
        a, b = rng.sample(range(5, 25), 2)
        if index == 2:
            add("math", "area", index, f"A rectangle is {a} m long. Its width is not given. What is its area in square meters?",
                f"Длина прямоугольника {a} м.\nШирина неизвестна.\nНайдите площадь в квадратных метрах.", None, reason=missing)
        else:
            add("math", "area", index, f"A rectangle is {a} m long and {b} m wide. What is its area in square meters?",
                f"Длина прямоугольника {a}\u202fм, ширина {b}\u202fм.\nНайдите площадь в квадратных метрах.", a*b)
        gap = 3 if index == 1 else 2
        add("math", "bounds", index, f"An integer must be greater than {-a} and less than {-a+gap}. What is the integer?",
            f"Целое число строго больше {str(-a).replace('-', '−')} и строго меньше {-a+gap}.\nКакое это число?",
            None if gap == 3 else -a+1, reason=missing if gap == 3 else None)
        received, shipped = 3, 4
        total = 2*b - received + b + shipped
        closing = total + received - shipped + (index == 2)
        clauses = ["Склады A и B.", f"В начале всего {total} ящиков.", f"Приход на склад A: {received}.",
                   f"Расход со склада B: {shipped}.", "После операций на складе A в 2 раза больше ящиков, чем на складе B.",
                   f"Итоговый остаток двух складов: {closing}."]
        rng.shuffle(clauses)
        add("math", "inventory", index, f"Two depots originally held {total} boxes in total. Depot A received {received} boxes "
            f"and depot B shipped {shipped}. After both movements A had twice as many boxes as B. "
            f"The final combined count was {closing}. How many boxes did depot B have before the movements?",
            "\n".join(clauses + ["Сколько ящиков было в начале на складе B?"]), None if index == 2 else b+shipped,
            reason=("logical_error", "the stated closing total disagrees with the transfers") if index == 2 else None)

        code = f"СВ-{a}{b}"
        add("search", "literal", index, "What is the Service code recorded in docs/notice.txt?",
            "Откройте `docs/notice.txt`.\nКакое значение указано в поле `Service code`?", code,
            assets={"docs/notice.txt": f"Service code: {code}\n"}, check="exact_str")
        name = ("Олег Миронов", "Анна Цой", "Kai Person")[index]
        add("search", "birth", index, f"According to kb/, In which year was {name} born?",
            f"В `kb/` найдите год рождения `{name}`.", str(1980+a),
            assets={"kb/person.md": markdown(name, {"Born": 1980+a})}, check="exact_str")
        rows = {"sales.csv": f"month,units\n1,{a}\n2,{b}\n3,{a+b}\n"}
        if index == 2:
            add("data_analysis", "filter", index, "In sales.csv, what is the average of units where month > 3?",
                "В файле `sales.csv` найдите среднее `units` по строкам, где `month` больше 3.", None,
                assets=rows, reason=("undefined_math", "no rows match"))
        else:
            op, word = (">=", "не меньше") if index == 0 else (">", "больше")
            add("data_analysis", "filter", index, f"In sales.csv, what is the sum of units where month {op} 2?",
                f"В файле `sales.csv` найдите сумму `units` по строкам, где `month` {word} 2.",
                a + 2*b if index == 0 else a+b, assets=rows)
    for suite in suites.values():
        suite.finish()


async def measure(root, output, model, budget, pause):
    from ping_floyd.proxy import Budget, Proxy, ProxyError

    key = os.environ["LLM_API_KEY"]
    oracle = json.loads((root / "english/oracle.json").read_text())
    events = []
    ledger = Budget(budget*10)
    proxy = Proxy(os.environ.get("LLM_BASE_URL", "https://api.neuraldeep.ru/v1"), key, model,
                  ledger, events.append, time.monotonic()+240)
    solve_prompt = "Solve the question. Reply with the integer only, or UNKNOWN if it has no unique answer."
    translate_prompt = ("Translate the full question into English. Keep every number, inequality, condition, and proper name. "
                        "Do not solve it. Return only the translated question.")

    async def call(key, phase, prompt, question, output_limit):
        if events and not ledger.blocked:
            await asyncio.sleep(pause)
        try:
            result = await proxy.invoke(f"{key}/{phase}", [{"role": "system", "content": prompt},
                                        {"role": "user", "content": question}], ledger.limit, output_limit)
            return result.strip()
        except ProxyError:
            return None

    async def compare(name):
        en, ru = [json.loads((root / language / "tasks" / name / "task.json").read_text())["instruction"]
                  for language in ("english", "russian")]
        expected = str(oracle[name]) if oracle[name] is not None else "UNKNOWN"
        answers = {}
        for phase, question in (("english", en), ("russian", ru)):
            answers[phase] = await call(name, phase, solve_prompt, question, 160)
        translated = await call(name, "translate", translate_prompt, ru, 300)
        answers["translated"] = await call(name, "translated", solve_prompt, translated, 160) if translated else None
        return {"task": name, "expected": expected, "answers": answers, "translation": translated}

    try:
        rows = [await compare(name) for name in sorted(oracle) if name.startswith("hard_math_")]
    finally:
        await proxy.client.aclose()
    totals = {}
    for variant, phases in {"english": {"english"}, "russian": {"russian"}, "translated": {"translate", "translated"}}.items():
        relevant = [e for e in events if e["task"].rsplit("/", 1)[1] in phases]
        usage = Counter()
        for event in relevant:
            usage.update({k: event.get(k, 0) for k in ("input_tokens", "output_tokens", "cost_tenths")})
        totals[variant] = dict(usage) | {"correct": sum(row["answers"][variant] == row["expected"] for row in rows),
                                        "tasks": len(rows), "calls": len(relevant)}
    dump(output, {"model": model, "budget_weighted_tokens": budget, "pause_s": pause, "totals": totals, "cases": rows,
                  "events": events, "unknown_reserved_tenths": ledger.reserved_unknown})
    print(json.dumps(totals))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate")
    create.add_argument("root", type=Path)
    create.add_argument("--seed", type=int, default=20261201)
    run = sub.add_parser("measure")
    run.add_argument("root", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--model", default="qwen3.8-27b-noreason")
    run.add_argument("--budget", type=int, default=4500)
    run.add_argument("--pause", type=float, default=2)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.root, args.seed)
    else:
        if args.budget <= 0 or not 0 <= args.pause <= 5:
            parser.error("budget must be positive and pause must be between zero and five seconds")
        asyncio.run(measure(args.root, args.output, args.model, args.budget, args.pause))


if __name__ == "__main__":
    main()
