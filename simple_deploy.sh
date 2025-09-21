#!/bin/bash

# 🚀 Simple Deploy Script
# Copy files to server manually

echo "📋 Files ready for server deployment:"
echo ""
echo "📄 server_index.php -> index.php (main website)"
echo "📄 server_api.php -> api.php (API endpoint)"
echo "📄 server_train.py -> train.py (Python trainer)"
echo ""
echo "🎯 Server path: /home/aiagent1/domains/analysis.aiagent.in.ua/public_html/"
echo ""
echo "📝 Commands to run on server:"
echo "cd /home/aiagent1/domains/analysis.aiagent.in.ua/public_html/"
echo "# Copy the content of server_* files to respective files on server"
echo "chmod +x train.py"
echo ""
echo "🌐 Test URL: https://analysis.aiagent.in.ua/"