#!/bin/bash

# 🌐 Setup Nginx with Subdomains
# Налаштування Nginx для роботи з субдоменами замість портів

echo "🌐 Setting up Nginx with subdomains..."
echo "========================================="

# Перевірити чи є права root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root or with sudo"
    echo "Usage: sudo ./setup-nginx.sh"
    exit 1
fi

# Директорії
NGINX_SITES_AVAILABLE="/etc/nginx/sites-available"
NGINX_SITES_ENABLED="/etc/nginx/sites-enabled"
PROJECT_DIR="/home/aiagent1/public_html/financial-ai-analyst"

# Створити директорії якщо не існують
mkdir -p "$NGINX_SITES_AVAILABLE"
mkdir -p "$NGINX_SITES_ENABLED"
mkdir -p "/var/log/nginx"

echo "📁 Setting up directories..."

# Скопіювати конфігурації
echo "📄 Installing Nginx configurations..."

# Конфігурація для analysis.aiagent.in.ua
cp "$PROJECT_DIR/nginx-config/analysis.aiagent.in.ua.conf" "$NGINX_SITES_AVAILABLE/"
ln -sf "$NGINX_SITES_AVAILABLE/analysis.aiagent.in.ua.conf" "$NGINX_SITES_ENABLED/"

# Конфігурація для deploy.aiagent.in.ua
cp "$PROJECT_DIR/nginx-config/deploy.aiagent.in.ua.conf" "$NGINX_SITES_AVAILABLE/"
ln -sf "$NGINX_SITES_AVAILABLE/deploy.aiagent.in.ua.conf" "$NGINX_SITES_ENABLED/"

# Видалити default конфігурацію якщо існує
rm -f "$NGINX_SITES_ENABLED/default"

# Тестувати конфігурацію Nginx
echo "🔍 Testing Nginx configuration..."
nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx configuration is valid"

    # Перезапустити Nginx
    echo "🔄 Restarting Nginx..."
    systemctl restart nginx
    systemctl enable nginx

    echo "✅ Nginx setup completed!"
    echo ""
    echo "🌐 Your subdomains are now configured:"
    echo "- https://analysis.aiagent.in.ua (LSTM Trading Assistant)"
    echo "- https://deploy.aiagent.in.ua (Auto-Deploy System)"
    echo ""
    echo "📊 Status check:"
    echo "- Analysis app: https://analysis.aiagent.in.ua/health"
    echo "- Deploy system: https://deploy.aiagent.in.ua/health"
    echo ""
    echo "⚠️  Make sure your DNS A records point to this server:"
    echo "- analysis.aiagent.in.ua → $(curl -s ifconfig.me || echo 'YOUR_SERVER_IP')"
    echo "- deploy.aiagent.in.ua → $(curl -s ifconfig.me || echo 'YOUR_SERVER_IP')"

else
    echo "❌ Nginx configuration test failed"
    echo "Please check the configuration files and try again"
    exit 1
fi

echo ""
echo "🎉 Subdomain setup complete!"
echo "No more ports in URLs - everything is clean and professional!"