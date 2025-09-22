
import sys
import os
import datetime
import random
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env файлу
load_dotenv()

# Додаємо шлях до lstm-trader, щоб можна було імпортувати predictor
sys.path.append(os.path.join(os.path.dirname(__file__), 'lstm-trader'))
from predictor import LSTMPredictor

app = Flask(__name__)
CORS(app)  # Дозволяємо запити з будь-якого джерела (для розробки)

@app.route('/api/predict/<string:pair>', methods=['GET'])
def predict(pair):
    '''
    Ендпоінт для отримання прогнозу для вказаної валютної пари.
    Використовує реальну модель для прогнозу.
    '''
    # Поки що використовуємо один таймфрейм D1 за замовчуванням
    timeframe = 'D1'
    predictor = LSTMPredictor()
    
    # Отримуємо прогноз
    result = predictor.predict(pair.upper(), timeframe)

    if 'error' in result:
        return jsonify({"success": False, "message": result['error']}), 404

    current_price = result['current_price']
    predicted_price = result['predicted_price']
    
    # Визначаємо рекомендовану дію та впевненість
    price_change = predicted_price - current_price
    change_percent = (price_change / current_price) * 100

    if change_percent > 0.5:
        action = "BUY"
        confidence = random.randint(75, 90) # Впевненість можна буде розраховувати складніше
    elif change_percent < -0.5:
        action = "SELL"
        confidence = random.randint(75, 90)
    else:
        action = "HOLD"
        confidence = random.randint(60, 80)

    # Формуємо фінальну відповідь
    response_data = {
        "success": True,
        "pair": pair.upper(),
        "current_price": round(current_price, 5),
        "predicted_price": round(predicted_price, 5),
        "price_change": round(price_change, 5),
        "price_change_percent": round(change_percent, 2),
        "action": action,
        "confidence": confidence,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return jsonify(response_data)

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Ендпоінт для перевірки стану сервісу.
    """
    return jsonify({
        "status": "online",
        "service": "Python LSTM Forex Predictor API",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    # Запуск Flask-сервера для локальної розробки
    # У продакшені буде використовуватись Gunicorn або інший WSGI-сервер
    app.run(host='0.0.0.0', port=5001, debug=True)
