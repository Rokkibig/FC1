#!/bin/bash
# This script automates the deployment of the LSTM Forex Predictor application on the server.

set -e # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting deployment..."

# Navigate to the project directory (update if your path is different)
PROJECT_DIR="/home/aiagent1/domains/analysis.aiagent.in.ua/public_html/FC1"
cd $PROJECT_DIR

echo "🔄 Pulling latest code from GitHub..."
git fetch origin main
git reset --hard origin/main

echo "🐍 Setting up Python environment..."
# Activate the virtual environment
# If you don't have one, you should create it first: python3 -m venv lstm_env
source lstm_env/bin/activate

# Install/update Python dependencies
pip install -r lstm-trader/requirements.txt

echo "⚙️ Setting up systemd service..."
# Copy the service file to the systemd directory
# This may require sudo privileges
sudo cp deploy/lstm-api.service /etc/systemd/system/lstm-api.service

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable lstm-api.service

echo "🔄 Restarting the API service..."
# Restart the service to apply changes
sudo systemctl restart lstm-api

# You might also need to configure your web server (e.g., Nginx) to proxy
# requests to the Gunicorn socket (lstm-api.sock).

echo "✅ Deployment finished successfully!"
