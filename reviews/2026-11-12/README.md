# Независимые регрессии из PR №1

`tests/test_review_tools.py` и `tests/test_review_workflow.py` перенесены побайтно из коммита `9cbdee9803e84f96e6a5a907539d817ae50edb8c`, [PR №1](https://github.com/PingFloydTeam/openJiuwen_Challenge_2026/pull/1). Это исходные 19 тестов автора, без изменения требований и без expectedFailure. Исторический отчёт и исходные логи остаются в PR: они описывают старый решатель `121f6c1`, а не текущий main.

Запуск из корня репозитория:

```sh
docker build --target bench -t ping-floyd:bench .
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=2g \
  --cap-drop ALL --security-opt no-new-privileges --cpus 2 --memory 4g --pids-limit 512 \
  --entrypoint python ping-floyd:bench \
  -m unittest discover -s tests -p 'test_review_*.py' -v
```

Нужен Linux с Landlock ABI ≥3 для положительных тестов файлового ввода-вывода. Mock HTTP заменяет только модель; настоящий openJiuwen, инструменты, запись ответов и расход остаются в проверке. Реальных API-вызовов нет. Полный текущий CI запускает эти тесты вместе с `bench/*checks.py`.

Проверяющий скрипт PR не дублируется: его задачи уже выполняют `bench.validation` и `bench.independent_coding`. Из рекомендаций дополнительно включён coding-пример в `bench.submission_smoke`: после агента код проверяется обычным unittest в отдельном контейнере, без собственного worker харнесса. Конкурсный образ не содержит тестов или fixtures.

Актуальные результаты и решения по замечаниям — в [отчёте интеграции](../../docs/pr1-integration-20261113.md).
