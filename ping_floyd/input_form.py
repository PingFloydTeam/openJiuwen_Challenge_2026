"""Conservative formatting and Russian-to-canonical question conversion."""
import re


FILE = r"[\w./-]+"
FIELD = r"[A-Za-z][\w ]*"
NUMBER = r"[+-]?\d+"
LITERAL = re.compile(r'(`[^`\n]*`|"[^"\n]*"|«[^»\n]*»|“[^”\n]*”|(?<!\w)\'[^\'\n]*\')')
HEADINGS = re.compile(r"^(?:#{1,3} +)?(?:Условие|Вопрос|Задача|Task|Question|Problem)\s*:\s*", re.I)
WRAPPER = re.compile(r"^(?:Пожалуйста[,! .]*|Спасибо[!.]*|Черновик для проверки\.|"
                     r"Please solve this\.|Thank you[!.]*)$", re.I)
COMPARISONS = {"не меньше": ">=", "не больше": "<=", "больше": ">", "меньше": "<", "равно": "="}
BALANCE_CLAUSES = {
    "depots": r"Склады ([A-Za-z][\w-]*) и ([A-Za-z][\w-]*)",
    "total": r"В начале всего (\d+) ящиков",
    "incoming": r"Приход на склад ([A-Za-z][\w-]*): (\d+)",
    "outgoing": r"Расход со склада ([A-Za-z][\w-]*): (\d+)",
    "ratio": r"После операций на складе ([A-Za-z][\w-]*) в ([23]) раза больше ящиков, чем на складе ([A-Za-z][\w-]*)",
    "closing": r"Итоговый остаток двух складов: (\d+)",
    "target": r"Сколько ящиков было в начале на складе ([A-Za-z][\w-]*)\?",
}


def unwrap_question(text):
    """Remove complete known wrapper lines. Keep unfamiliar content verbatim."""
    if len(text) > 16_000:
        return text
    # Unmatched delimiters may belong to multiline strings or code. Leave them intact.
    if re.search(r'[`"«»“”]|(?<!\w)\'', LITERAL.sub("", text)):
        return text
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("```", "~~~")):
            return text
        line = re.sub(r"^• +(?=[A-Za-zА-Яа-яЁё])", "", line)
        line = HEADINGS.sub("", line)
        if line.startswith("**") and line.endswith("**") and line.count("**") == 2:
            line = line[2:-2].strip()
        if line and not WRAPPER.fullmatch(line):
            lines.append(line)
    return " ".join(lines)


def russian_balance(text):
    if "склад" not in text.casefold():
        return None
    parts = {}
    clauses = text.rstrip(".").split(". ")
    if len(clauses) not in {6, 7}:
        return None
    for clause in clauses:
        for key, pattern in BALANCE_CLAUSES.items():
            if match := re.fullmatch(pattern, clause, re.I):
                if key in parts:
                    return None
                parts[key] = match.groups()
                break
        else:
            return None
    if not (BALANCE_CLAUSES.keys() - {"closing"} <= parts.keys()):
        return None
    first, received = parts["incoming"]
    second, shipped = parts["outgoing"]
    target, = parts["target"]
    large, ratio, small = parts["ratio"]
    names = {first.casefold(), second.casefold()}
    if (len(names) != 2 or names != {label.casefold() for label in parts["depots"]}
            or (large.casefold(), small.casefold()) != (first.casefold(), second.casefold())
            or target.casefold() not in names):
        return None
    closing = f"The final combined count was {parts['closing'][0]}. " if "closing" in parts else ""
    factor = "twice" if ratio == "2" else "three times"
    return (f"Two depots originally held {parts['total'][0]} boxes in total. Depot {first} received {received} boxes "
            f"and depot {second} shipped {shipped}. After both movements {first} had {factor} as many boxes as {second}. "
            f"{closing}How many boxes did depot {target} have before the movements?")


def russian_math(question):
    # Only convert units and operators after the entire question matches a grammar.
    text = question.translate(str.maketrans({"\u00a0": " ", "\u202f": " ", "−": "-"}))
    text = re.sub(r" {2,}", " ", text)
    if balance := russian_balance(text):
        return balance
    if match := re.fullmatch(
            r"(?:Длина прямоугольника (\d+) м, ширина (\d+) м\. |"
            r"У прямоугольника длина (\d+) м и ширина (\d+) м\. )"
            r"(?:Найдите|Вычислите) площадь в квадратных метрах[.?]", text, re.I):
        a, b = match.groups()[:2] if match[1] is not None else match.groups()[2:]
        return f"A rectangle is {a} m long and {b} m wide. What is its area in square meters?"
    if match := re.fullmatch(r"Длина прямоугольника (\d+) м\. Ширина неизвестна\. "
                             r"(?:Найдите|Вычислите) площадь в квадратных метрах[.?]", text, re.I):
        return f"A rectangle is {match[1]} m long. Its width is not given. What is its area in square meters?"
    if match := re.fullmatch(rf"(?:Найдите|Вычислите) среднее арифметическое чисел ({NUMBER}), ({NUMBER}) и ({NUMBER})[.?]",
                             text, re.I):
        a, b, c = (int(value) for value in match.groups())
        return f"The three measured values are {a}, {b}, and {c}. Find their arithmetic mean."
    if match := re.fullmatch(rf"Целое число строго больше ({NUMBER}) и строго меньше ({NUMBER})\. "
                             r"Какое это число\?", text, re.I):
        return f"An integer must be greater than {int(match[1])} and less than {int(match[2])}. What is the integer?"
    if match := re.fullmatch(r"(?:Вычислите|Посчитайте) ([0-9().+*/% ×÷-]+)[.?]", text, re.I):
        return f"Calculate {match[1]}?"
    return None


def canonical_question(question, kind):
    if kind == "math":
        return russian_math(question)
    if kind == "search":
        if match := re.fullmatch(rf"Откройте `({FILE}\.txt)`\. Какое значение указано в поле `({FIELD})`\?", question, re.I):
            return f"What is the {match[2]} recorded in {match[1]}?"
        if match := re.fullmatch(r"В `kb/` найдите год рождения `([\w .-]+)`[.?]", question, re.I):
            return f"According to kb/, In which year was {match[1]} born?"
    if kind == "data_analysis":
        if match := re.fullmatch(
                rf"В файле `({FILE}\.(?:csv|tsv))` найдите (сумму|среднее) `({FIELD})` по строкам, где "
                rf"`({FIELD})` (не меньше|не больше|больше|меньше|равно) ({NUMBER})[.?]", question, re.I):
            file, operation, value, field, comparator, threshold = match.groups()
            operation = "sum" if operation.casefold() == "сумму" else "average"
            if " " not in value and " " not in field:
                return f"In {file}, what is the {operation} of {value} where {field} {COMPARISONS[comparator.casefold()]} {int(threshold)}?"
    return None
