# ⚡ Швидке Налаштування DirectAdmin Git Деплою

## 🎯 Мета: git push → автодеплой за 3 хвилини!

### 1️⃣ Налаштувати Git в DirectAdmin

1. **Увійти в DirectAdmin панель:**
   - Відкрити: https://tzk302.nic.ua:2222
   - Логін: aiagent1
   - Пароль: 62dmtml2eqr

2. **Налаштувати Git репозиторій:**
   - Натиснути **"GIT"** в панелі
   - Вибрати піддомен: **analysis.aiagent.in.ua**
   - Repository URL: **https://github.com/Rokkibig/FC1.git**
   - Branch: **main**
   - Path: **public_html/analysis.aiagent.in.ua**
   - Увімкнути **"Auto Pull"**

3. **Зберегти налаштування**

### 2️⃣ Додати Python startup скрипт

Створити файл `.htaccess` в директорії `lstm-trader/`:

```apache
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ /cgi-bin/python3 app.py [L]
```

### 3️⃣ Тест деплою

```bash
# Локально:
git push origin main

# Перевірити в GitHub Actions що деплой пройшов успішно
# Відкрити: https://analysis.aiagent.in.ua
```

## ✅ Готово!

Тепер кожен `git push` автоматично синхронізується з сервером через DirectAdmin Git!

## 🔧 Переваги DirectAdmin Git

✅ **Простота** - без SSH ключів та секретів
✅ **Автоматичність** - git push → auto sync
✅ **Безпека** - DirectAdmin керує доступом
✅ **Надійність** - вбудована інтеграція

## 🔍 Якщо щось не працює

### Перевірити DirectAdmin Git статус
1. Зайти в DirectAdmin → GIT
2. Перевірити статус sync
3. Подивитись логи помилок

### Перевірити сайт
```bash
curl https://analysis.aiagent.in.ua/health
```

### Мануальний sync в DirectAdmin
1. DirectAdmin → GIT → analysis.aiagent.in.ua
2. Натиснути "Pull Now"

## 🎉 Результат

**git push** → DirectAdmin Git Sync → **https://analysis.aiagent.in.ua**

**Найпростіший професійний деплой готовий!** 🚀