#!/bin/bash

# 🚀 LSTM Forex Predictor - Deploy to Server
# Чистий деплой із GitHub

echo "🎯 Деплой на сервер: analysis.aiagent.in.ua"
echo ""
echo "📂 Перехід в папку сайту:"
echo "cd /home/aiagent1/domains/analysis.aiagent.in.ua/public_html/"
echo ""
echo "🔄 Оновлення з GitHub:"
echo "git fetch origin main"
echo "git reset --hard origin/main"
echo ""
echo "📝 Перейменування файлів для продакшену:"
echo "cp server_index.php index.php"
echo "cp server_api.php api.php"
echo "cp server_train.py train.py"
echo "chmod +x train.py"
echo ""
echo "🧪 Тестування API:"
echo "python3 train.py EURUSD"
echo "curl 'https://analysis.aiagent.in.ua/api.php?action=health'"
echo ""
echo "🌐 Готово! Сайт доступний: https://analysis.aiagent.in.ua/"