#!/bin/bash

# 🚀 One-time Server Setup
# Налаштування сервера для роботи з GitHub Actions деплоєм

echo "🚀 Setting up server for GitHub Actions deploy..."
echo "================================================="

# Перевірити чи виконується як правильний користувач
CURRENT_USER=$(whoami)
echo "👤 Current user: $CURRENT_USER"

# Створити робочу директорію
WORK_DIR="/home/aiagent1/public_html/financial-ai-analyst"
echo "📁 Creating work directory: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Клонувати репозиторій якщо не існує
if [ ! -d ".git" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/Rokkibig/FC1.git .
else
    echo "📥 Updating repository..."
    git pull origin main
fi

# Налаштувати Nginx
echo "🌐 Setting up Nginx..."
if [ -f "setup-nginx.sh" ]; then
    sudo ./setup-nginx.sh
else
    echo "⚠️ Nginx setup script not found, skipping..."
fi

# Перевірити Python
echo "🐍 Checking Python..."
python3 --version
which python3

# Створити простий systemd сервіс (опціонально)
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/lstm-trader.service > /dev/null << EOF
[Unit]
Description=LSTM Trading Assistant
After=network.target

[Service]
Type=simple
User=aiagent1
WorkingDirectory=$WORK_DIR/lstm-trader
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Активувати сервіс
sudo systemctl daemon-reload
sudo systemctl enable lstm-trader

echo "✅ Server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Add SSH public key to ~/.ssh/authorized_keys"
echo "2. Set GitHub Secrets:"
echo "   - SERVER_HOST=analysis.aiagent.in.ua"
echo "   - SERVER_USER=aiagent1"
echo "   - SSH_PRIVATE_KEY=(your private key)"
echo ""
echo "3. Test deployment:"
echo "   git push origin main"
echo ""
echo "🌐 App will be available at: https://analysis.aiagent.in.ua"
echo "🎉 Ready for GitHub Actions deployment!"