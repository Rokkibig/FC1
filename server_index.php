<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 LSTM Forex Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
        }
        .container {
            background: rgba(255,255,255,0.95); backdrop-filter: blur(10px);
            border-radius: 20px; padding: 40px; box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            text-align: center; max-width: 500px; width: 90%;
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 1.1em; }
        .pair-buttons { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 30px; }
        .pair-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049); color: white; border: none;
            padding: 15px; border-radius: 10px; font-size: 16px; font-weight: bold;
            cursor: pointer; transition: all 0.3s ease;
        }
        .pair-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(76,175,80,0.4); }
        .pair-btn.active { background: linear-gradient(45deg, #2196F3, #1976D2); }
        .predict-btn {
            background: linear-gradient(45deg, #FF6B6B, #FF5252); color: white; border: none;
            padding: 20px 40px; border-radius: 15px; font-size: 18px; font-weight: bold;
            cursor: pointer; transition: all 0.3s ease; margin-bottom: 20px;
        }
        .predict-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255,107,107,0.4); }
        .predict-btn:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        .result {
            margin-top: 20px; padding: 20px; border-radius: 15px;
            background: linear-gradient(45deg, #f8f9fa, #e9ecef);
            border-left: 5px solid #4CAF50; text-align: left;
        }
        .loading { display: none; margin: 20px 0; }
        .spinner {
            border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%;
            width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .price { font-size: 24px; font-weight: bold; color: #2196F3; }
        .change.positive { color: #4CAF50; }
        .change.negative { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 LSTM Forex</h1>
        <p class="subtitle">Штучний інтелект для прогнозування валют</p>

        <div class="pair-buttons">
            <button class="pair-btn active" onclick="selectPair('EURUSD')">EUR/USD</button>
            <button class="pair-btn" onclick="selectPair('GBPUSD')">GBP/USD</button>
            <button class="pair-btn" onclick="selectPair('USDJPY')">USD/JPY</button>
            <button class="pair-btn" onclick="selectPair('AUDUSD')">AUD/USD</button>
            <button class="pair-btn" onclick="selectPair('USDCHF')">USD/CHF</button>
            <button class="pair-btn" onclick="selectPair('USDCAD')">USD/CAD</button>
        </div>

        <button class="predict-btn" onclick="getPrediction()" id="predictBtn">
            📈 Отримати прогноз
        </button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Аналізую ринок...</p>
        </div>

        <div id="result"></div>
    </div>

    <script>
        let selectedPair = 'EURUSD';

        function selectPair(pair) {
            selectedPair = pair;
            document.querySelectorAll('.pair-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }

        async function getPrediction() {
            const predictBtn = document.getElementById('predictBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            predictBtn.disabled = true;
            loading.style.display = 'block';
            result.innerHTML = '';

            try {
                const response = await fetch(`api.php?action=predict&pair=${selectedPair}`);
                const data = await response.json();

                if (data.success) {
                    displayResult(data);
                } else {
                    displayError(data.message || 'Помилка отримання прогнозу');
                }
            } catch (error) {
                displayError('Помилка з\'єднання з сервером');
            } finally {
                predictBtn.disabled = false;
                loading.style.display = 'none';
            }
        }

        function displayResult(data) {
            const result = document.getElementById('result');
            const changeClass = data.price_change > 0 ? 'positive' : 'negative';
            const changeSymbol = data.price_change > 0 ? '↗️' : '↘️';

            result.innerHTML = `
                <div class="result">
                    <h3>📊 Прогноз для ${data.pair}</h3>
                    <div style="margin: 15px 0;">
                        <div>Поточна ціна: <span class="price">${data.current_price}</span></div>
                        <div>Прогноз на 24г: <span class="price">${data.predicted_price}</span></div>
                        <div>Зміна: <span class="change ${changeClass}">
                            ${changeSymbol} ${data.price_change > 0 ? '+' : ''}${data.price_change_percent}%
                        </span></div>
                    </div>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                        <div><strong>📈 Рекомендація:</strong> ${data.action}</div>
                        <div><strong>🎯 Довіра:</strong> ${data.confidence}%</div>
                        <div><strong>📅 Оновлено:</strong> ${data.timestamp}</div>
                    </div>
                </div>
            `;
        }

        function displayError(message) {
            const result = document.getElementById('result');
            result.innerHTML = `
                <div class="result" style="border-left-color: #f44336;">
                    <h3>❌ Помилка</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    </script>
</body>
</html>