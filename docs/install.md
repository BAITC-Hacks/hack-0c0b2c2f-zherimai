# Подготовка окружения DalaAI

Требуются **Python 3.13**, обычный ноутбук и три предоставленных parquet в `data/`. Версии зависимостей закреплены в `requirements.txt`. GPU, сервер, API-ключи и переменные окружения не нужны. `.env.example` документирует отсутствие обязательных настроек.

Создавать `.env` для основного запуска не требуется; приложение не загружает этот файл автоматически. Переменные `OPENAI_API_KEY` и `OPENAI_MODEL` задаются только для явно включённого опционального AI-планировщика по [отдельной инструкции](ai-assistant.md).

### Если Python 3.13 ещё не установлен

Установите **uv** по [официальной инструкции Astral](https://docs.astral.sh/uv/getting-started/installation/) — самостоятельный установщик не требует предварительной установки Python. Ниже проверен uv **0.12.18**. В свежем клоне из корня репозитория, macOS/Linux:

```bash
uv venv --managed-python --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt && .venv/bin/python run.py --data data --out out
```

`--managed-python` выбирает Python, установленный uv; при его отсутствии uv скачивает подходящий 3.13 независимо от версии системного Python. [Описание управления Python](https://docs.astral.sh/uv/guides/install-python/).

Windows PowerShell, после установки uv:

```powershell
uv venv --managed-python --python 3.13 .venv
if ($LASTEXITCODE -ne 0) { throw 'Не удалось подготовить Python 3.13' }
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости' }
.\.venv\Scripts\python.exe run.py --data data --out out
```

**Этап подготовки требует интернета** для загрузки uv, Python и пакетов. Затем запускайте `.venv/bin/python run.py` напрямую — uv и сеть для пересчёта не нужны. Это не установка из пустой машины без сети; для полностью автономной машины Python и совместимые пакеты готовятся заранее по разделу ниже.

Проверено на macOS ARM64 из копии без Python-окружения и out: скачан CPython **3.13.15**, установлены все закреплённые пакеты, холодный расчёт **15,009 с**, 90 тестов и аудит PASS; 7 CSV и HTML совпали побайтно. Установка Windows/Linux этим прогоном не подтверждается. [Протокол uv](uv-verification.md).

### Если Python 3.13 уже установлен

Из корня репозитория, macOS/Linux, первая установка и полный пересчёт одной строкой:

```bash
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python run.py --data data --out out
```

Первоначальная установка зависимостей требует доступа к источнику пакетов. После установки полный пересчёт работает без интернета:

```bash
.venv/bin/python run.py --data data --out out
```

Откройте `out/network.html` в браузере двойным щелчком. Страница содержит данные и код интерфейса локально. Полный пересчёт включает CSV, дополнительные отчёты и страницу сети. Фактическое время текущего запуска сохраняется в `out/report.json`; установка зависимостей в это время не входит.

Windows PowerShell, Python 3.13:

```powershell
py -3.13 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение Python 3.13' }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости' }
.\.venv\Scripts\python.exe run.py --data data --out out
if ($LASTEXITCODE -ne 0) { throw 'Расчёт завершился ошибкой' }
Start-Process .\out\network.html
```

Для Python другой версии используйте путь uv выше либо отдельно установите 3.13. Проверьте `python3.13 --version` / `py -3.13 --version`. Совместимость пакетов с Python 3.12 не является проверкой приложения: поддержка 3.12 не заявляется. Версии пакетов не изменяйте.

### Установка без интернета

На машине с интернетом с **той же ОС, архитектурой и версией Python**, что у проверяющего, подготовьте пакеты:

```bash
python3.13 -m pip download --only-binary=:all: -r requirements.txt --dest wheelhouse
```

Перенесите репозиторий вместе с `wheelhouse/` на проверяющую машину и выполните:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
.venv/bin/python run.py --data data --out out
```

Windows PowerShell: сначала на подключённой к интернету Windows-машине с той же архитектурой и Python 3.13 подготовьте wheels:

```powershell
py -3.13 -m pip download --only-binary=:all: -r requirements.txt --dest wheelhouse
```

Перенесите репозиторий и `wheelhouse/` на проверяющую Windows-машину с уже установленным Python 3.13. Затем, без сети:

```powershell
py -3.13 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение Python 3.13' }
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить пакеты из wheelhouse' }
.\.venv\Scripts\python.exe run.py --data data --out out
```

Если wheel для целевой платформы отсутствует, подготовка завершится ошибкой; нельзя подменять её обещанием работы в произвольной среде. `wheelhouse` и установщик Python не включены в репозиторий. Отдельная чистая машина без заранее установленного Python и без подготовленных wheels не сможет установить проект офлайн.


Вернуться к [README и основному сценарию](../README.md#быстрый-запуск).
