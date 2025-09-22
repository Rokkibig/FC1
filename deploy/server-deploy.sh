#!/bin/bash

# 🚀 Manual deploy script for server
# Запустіть цей скрипт на сервері для ручного деплою

echo "🚀 Manual Deploy: LSTM Trading Assistant"
echo "========================================"

# Перейти до робочої директорії
cd /home/aiagent1/public_html/financial-ai-analyst

# Зупинити поточні сервіси
echo "🛑 Stopping current services..."
pkill -f "python.*app.py" || echo "No running services found"
pkill -f "python.*simple_server.py" || echo "No simple server found"
sleep 2

# Клонувати або оновити репозиторій
echo "📥 Getting latest code..."
if [ -d ".git" ]; then
    echo "Updating existing repository..."
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/Rokkibig/FC1.git .
fi

# Перейти до LSTM директорії
cd lstm-trader/

# Зробити скрипт виконуваним
chmod +x deploy.sh

# Запустити автодеплой
echo "🚀 Running auto-deploy script..."
./deploy.sh

echo ""
echo "🎉 Manual deployment completed!"
echo "📊 LSTM Trading Assistant should now be running"
echo "🌐 Check: https://analysis.aiagent.in.ua"
echo "❤️ Health: https://analysis.aiagent.in.ua/health"