# 🚀 Автоматичний деплой Financial AI Analyst

Цей документ описує як налаштувати автоматичний деплой одною командою `git push`.

## 📋 Огляд

Після налаштування, кожен `git push` на branch `main` автоматично:
1. ✅ Запускає тести
2. 🏗️ Будує Docker образи
3. 🚀 Деплоїть на сервер analysis.aiagent.in.ua
4. 🔍 Перевіряє здоров'я системи

## 🛠️ Налаштування (виконати один раз)

### Крок 1: Налаштування SSH ключів

```bash
# Запустити скрипт для генерації SSH ключів
chmod +x deployment/ssh/setup_ssh_keys.sh
./deployment/ssh/setup_ssh_keys.sh
```

Скрипт покаже публічний ключ, який потрібно додати в DirectAdmin:

1. Відкрити [DirectAdmin](https://analysis.aiagent.in.ua:2222)
2. Увійти як `aiagent1` / `62dmtml2eqr`
3. Перейти до **Розширені налаштування** → **Ключі SSH**
4. Натиснути **Додати ключ**
5. Вставити публічний ключ та зберегти

### Крок 2: Налаштування GitHub Secrets

Додати ці секрети в GitHub репозиторій (`Settings` → `Secrets and variables` → `Actions`):

```
VPS_SSH_KEY=<приватний SSH ключ>
VPS_HOST=65.108.39.53
VPS_USER=aiagent1
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/financial_ai
REDIS_URL=redis://redis:6379/0
SECRET_KEY=<випадковий безпечний ключ>
OPENAI_API_KEY=<ваш OpenAI ключ>
ANTHROPIC_API_KEY=<ваш Anthropic ключ>
ALPHA_VANTAGE_API_KEY=<ваш Alpha Vantage ключ>
NEWSAPI_KEY=<ваш NewsAPI ключ>
ALERT_EMAIL=admin@analysis.aiagent.in.ua
```

### Крок 3: Налаштування Webhook (опціонально)

Для миттєвого деплою через webhook:

```bash
# На сервері виконати
scp deployment/webhook/setup_webhook.sh aiagent1@65.108.39.53:~/
ssh aiagent1@65.108.39.53 "chmod +x ~/setup_webhook.sh && ~/setup_webhook.sh"
```

Потім додати webhook в GitHub:
- URL: `http://analysis.aiagent.in.ua:9000/webhook`
- Content type: `application/json`
- Secret: встановити в налаштуваннях webhook сервера

## 🚀 Використання

### Автоматичний деплой через GitHub Actions

```bash
# Просто зробити commit і push
git add .
git commit -m "Нова функція або виправлення"
git push origin main

# GitHub Actions автоматично:
# 1. Запустить тести
# 2. Побудує Docker образи
# 3. Задеплоїть на сервер
# 4. Перевірить здоров'я системи
```

### Ручний деплой (якщо потрібно)

```bash
# Підключитися до сервера
ssh aiagent1@65.108.39.53

# Запустити деплой
cd /home/aiagent1/domains/assistant.aiagent.in.ua/webhook
./deploy.sh
```

## 📊 Моніторинг

### Перевірка статусу деплою

- **GitHub Actions**: https://github.com/Rokkibig/FC1/actions
- **Додаток**: http://analysis.aiagent.in.ua/health
- **API**: http://analysis.aiagent.in.ua/api/v1/monitoring/health

### Логи на сервері

```bash
# Логи деплою
ssh aiagent1@65.108.39.53 "tail -f /home/aiagent1/domains/analysis.aiagent.in.ua/logs/deploy.log"

# Логи webhook сервера (якщо використовується)
ssh aiagent1@65.108.39.53 "journalctl -u financial-ai-webhook -f"

# Логи додатку
ssh aiagent1@65.108.39.53 "cd /home/aiagent1/domains/analysis.aiagent.in.ua/financial-ai-analyst && docker-compose logs -f"
```

## 🔧 Налаштування середовища

### Файл .env на сервері

Автоматично створюється при деплої з наступними налаштуваннями:

```env
# База даних
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/financial_ai
REDIS_URL=redis://redis:6379/0

# Безпека
SECRET_KEY=<безпечний ключ з GitHub Secrets>

# API ключі
OPENAI_API_KEY=<з GitHub Secrets>
ANTHROPIC_API_KEY=<з GitHub Secrets>
ALPHA_VANTAGE_API_KEY=<з GitHub Secrets>
NEWSAPI_KEY=<з GitHub Secrets>

# Налаштування додатку
APP_NAME=Financial AI Analyst
DEBUG=False
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Торгівля
DEFAULT_SYMBOLS=AAPL,GOOGL,MSFT,TSLA,SPY
TRADING_SIMULATION=True
INITIAL_CAPITAL=100000

# Моніторинг
ALERT_EMAIL=admin@analysis.aiagent.in.ua
```

## 🛡️ Безпека

- SSH ключі використовуються для автентифікації
- API ключі зберігаються в GitHub Secrets
- Webhook підписується секретним ключем
- HTTPS буде налаштований автоматично

## 🔄 Процес деплою

1. **Push на main** → GitHub Actions запускається
2. **Тестування** → Запуск всіх тестів з PostgreSQL/Redis
3. **Збірка** → Створення Docker образів
4. **Деплой** → Підключення до VPS через SSH
5. **Оновлення** → Скачування коду, збірка контейнерів
6. **Перевірка** → Health check та ініціалізація БД
7. **Завершення** → Очищення старих образів

## 🚨 Вирішення проблем

### Деплой не працює

```bash
# Перевірити статус GitHub Actions
# https://github.com/Rokkibig/FC1/actions

# Перевірити SSH з'єднання
ssh aiagent1@65.108.39.53

# Перевірити логи
ssh aiagent1@65.108.39.53 "tail -100 /home/aiagent1/domains/analysis.aiagent.in.ua/logs/deploy.log"
```

### Контейнери не запускаються

```bash
# Підключитися до сервера
ssh aiagent1@65.108.39.53

# Перейти до папки додатку
cd /home/aiagent1/domains/analysis.aiagent.in.ua/financial-ai-analyst

# Перевірити статус
docker-compose ps

# Перегляд логів
docker-compose logs

# Перезапуск
docker-compose down && docker-compose up -d
```

### База даних не працює

```bash
# Перевірка контейнера PostgreSQL
docker-compose exec postgres psql -U postgres -d financial_ai -c "\dt"

# Ініціалізація БД
docker-compose exec api python -c "
import asyncio
from src.data_pipeline.storage.database import init_database
asyncio.run(init_database())
"
```

## 📞 Підтримка

При виникненні проблем:
1. Перевірити логи GitHub Actions
2. Перевірити логи на сервері
3. Перевірити статус контейнерів
4. Перевірити health endpoints

**Готово! Тепер одна команда `git push` автоматично деплоїть ваш код! 🎉**