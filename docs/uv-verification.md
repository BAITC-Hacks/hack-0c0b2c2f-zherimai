# Проверка установки Python 3.13 через uv

Дата: 2026-09-23T16:17:29.518215+05:00. Среда: macOS 27.0 ARM64.

Источник: официальный commit `8b36d9994a4f3992188b12175ae58a1e51a9f7e0` плюс рабочие UTF-8 правки интегратора в `run.py`, `aml/validate.py`, `scripts/analyze.py`. Копия создана из `git archive` без `.git`, `.venv` и `out`. Отслеживаемые файлы исходного репозитория не изменялись.

## Подготовка с интернетом

Исходно uv отсутствовал. Bootstrap venv создан системным Python 3.13.2 только для установки uv. Глобальный Python, пользовательские настройки и существующее рабочее окружение не менялись.

```sh
python3 -m venv /private/tmp/hackalem-uv-verification-pgdk2l6v/bootstrap
/private/tmp/hackalem-uv-verification-pgdk2l6v/bootstrap/bin/python -m pip install --index-url https://pypi.org/simple uv
uv venv --managed-python --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Использован uv 0.12.18. `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PYTHON_BIN_DIR` направлены в отдельную пустую временную папку; `UV_NO_CONFIG=1`. Флаг `--managed-python` гарантировал реальную загрузку CPython вместо выбора уже имеющегося системного интерпретатора. Скачан CPython **3.13.15**, затем все семь закреплённых библиотек. Их версии совпали с `requirements.txt`.

Подготовка: bootstrap venv 2,491 с; установка uv 3,486 с; Python + venv 3,760 с; библиотеки 7,555 с. Интернет нужен для этого этапа; установка с нуля полностью без интернета не заявляется.

## Расчёт после подготовки

Следующие команды выполнены в стандартной ограниченной среде, без `OPENAI_*`, с `UV_OFFLINE=1` и `PIP_NO_INDEX=1`. Повторных загрузок и платных API-вызовов не было.

```sh
uv pip check --python .venv/bin/python --offline
.venv/bin/python run.py --data data --out out
.venv/bin/python validate.py --data data --out out
.venv/bin/python -W error::ResourceWarning -m unittest discover -s tests -v
.venv/bin/python scripts/audit_top.py --data data --out out
```

| Проверка | Время процесса | Результат |
|---|---:|---|
| Пакеты | 0.025 с | PASS |
| Холодный расчёт | 15.009 с | PASS |
| Валидатор | 0.861 с | PASS |
| Тесты | 4.303 с | PASS |
| Независимый аудит | 1.633 с | PASS |

Внутреннее время расчёта: 14.5563 с. **90 тестов PASS**, аудит: `mismatches={}`. 2 248 узлов, 49 групп, топ-50; роли: 21 coordinator / 42 consolidator / 43 distributor / 36 transit / 566 terminal / 1 540 peripheral.

Все 7 CSV и HTML **побайтно совпали** с отслеживаемыми результатами main:

| Файл | SHA-256 |
|---|---|
| `clusters.csv` | `202688cd717db8a5ef28fc5e054269ed5d019213ec9eaba2ede0c2c4e796bdcb` |
| `data_requests.csv` | `4eccacf3d4f78e5dd047e42b2889591c2866db8aa8726d93b3e4e46d151bc593` |
| `node_stability.csv` | `869cd7b19e193b3fd825c6cb5bf406d211d4670eabd7406f2df4f16eb294294d` |
| `nodes_roles.csv` | `62302a2c070b46fd5aa22af80a916ce3a5404d6b918ae1bc252b66458cecfe4e` |
| `resilience.csv` | `1f62c07b8374c216cacde99a1746dc8f1eaef5ac31536b6a3b342d4b5d69caf2` |
| `sensitivity.csv` | `e8e5dd362815f84af7992113a6ac5a4b78f43c3066de32bab847206a46af6858` |
| `top_nodes.csv` | `ffe2e8be5b05195682a380817683ecd17d0fddaa91a21f18b38a4fac07e2998e` |
| `network.html` | `256876e3773855ff66bc543a0d6dd6ffdca360171d6abc64cecaf5de31d5bef3` |

## Требование к Python

Метаданные установленных пакетов: networkx 3.7 — `!=3.14.1,>=3.12`; numpy 2.5.3 и scipy 1.18.1 — `>=3.12`; pandas 3.0.6 — `>=3.11`; pyarrow 25.0.1 — `>=3.10`. Совместимость зависимостей с 3.12 не заменяет испытание приложения. Для инструкции оставляем **проверенный Python 3.13**.

## Ограничения

- Новый полный uv-путь проверен только на macOS ARM64; не подтверждает новую установку на Windows/Linux.
- Python 3.12 не проверялся.
- Node, использованный тестами интерфейса, был доступен заранее; для расчёта run.py он не нужен.
- Визуальная проверка браузера в эту проверку не входила.
- `report.json` содержит версию Python и время, поэтому не включён в сравнение детерминированных файлов.

Переносимый протокол: [uv-verification.json](uv-verification.json). Подробные команды и журналы интегратора сохранены локально в игнорируемом `out-check` и временной папке; они не нужны для запуска из README.
