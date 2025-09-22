# 🌐 Налаштування Субдоменів

Професійна конфігурація з субдоменами замість портів!

## 🎯 Мета

**Замість:** `http://analysis.aiagent.in.ua:8001`
**Тепер:** `https://analysis.aiagent.in.ua` ✨

## 📋 Структура Субдоменів

- **`analysis.aiagent.in.ua`** - LSTM Trading Assistant
- **`deploy.aiagent.in.ua`** - Система автодеплою
- **`api.aiagent.in.ua`** - API endpoints (резерв)

## 🚀 Одноразове Налаштування

### 1. DNS Налаштування (в панелі домену)

Додайте A записи:
```
analysis.aiagent.in.ua  →  YOUR_SERVER_IP
deploy.aiagent.in.ua    →  YOUR_SERVER_IP
```

### 2. Nginx Setup на Сервері

```bash
# На сервері виконайте:
cd /home/aiagent1/public_html/financial-ai-analyst

# Завантажити останні файли
git pull origin main

# Встановити Nginx конфігурацію
sudo ./setup-nginx.sh
```

### 3. Запустити Систему

```bash
# Запустити webhook систему
./start-webhook.sh

# Перевірити статус
curl https://deploy.aiagent.in.ua/health
curl https://analysis.aiagent.in.ua/health
```

### 4. GitHub Webhook

- **URL:** `https://deploy.aiagent.in.ua/webhook`
- **Content type:** `application/json`
- **Events:** Just the push event

## ✨ Результат

### До:
- ❌ `http://analysis.aiagent.in.ua:8001`
- ❌ `http://analysis.aiagent.in.ua:3000/webhook`

### Після:
- ✅ `https://analysis.aiagent.in.ua`
- ✅ `https://deploy.aiagent.in.ua`

## 🔧 Технічні Деталі

### Nginx Reverse Proxy
- Nginx слухає порт 80/443
- Проксує запити на внутрішні порти
- SSL термінація (готово до Let's Encrypt)
- Логування та кешування

### Внутрішня Архітектура
```
Internet → Nginx (80/443) → Local Services
                    ↓
    analysis.aiagent.in.ua → localhost:8001 (LSTM App)
    deploy.aiagent.in.ua   → localhost:3000 (Webhook)
```

### Переваги
- 🌐 Чисті URL без портів
- 🔒 Готовність до SSL
- 📊 Централізоване логування
- ⚡ Кращі перформанси
- 🛡️ Додаткова безпека

## 📊 Моніторинг

- **LSTM App:** https://analysis.aiagent.in.ua
- **Deploy System:** https://deploy.aiagent.in.ua
- **Health Checks:**
  - https://analysis.aiagent.in.ua/health
  - https://deploy.aiagent.in.ua/health

## 🔧 Troubleshooting

### Перевірити Nginx
```bash
sudo nginx -t
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

### Перевірити DNS
```bash
nslookup analysis.aiagent.in.ua
nslookup deploy.aiagent.in.ua
```

### Перевірити Сервіси
```bash
ps aux | grep python
netstat -tlnp | grep :8001
netstat -tlnp | grep :3000
```

## 🎉 Готово!

Тепер у вас професійні субдомени без портів! 🚀

**Workflow:** code → git push → автодеплой на `https://analysis.aiagent.in.ua`