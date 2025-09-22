#!/usr/bin/env python3
"""
🔮 Real LSTM Model Predictor
Loads a trained model from an SQLite database and makes predictions.
"""

import os
import numpy as np
import pandas as pd
import sqlite3
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env файлу
load_dotenv()

# Assuming model_trainer.py is in the same directory or accessible
# We need the feature preparation logic from it.
from model_trainer import LSTMForexTrainer

class LSTMPredictor:
    def __init__(self, db_file="forex_data.db"):
        """Initialize the predictor with the SQLite database file"""
        self.db_file = db_file
        self.models_dir = 'models'
        self.scalers_dir = 'scalers'
        self.sequence_length = 60  # Should be same as trainer

        # We need an instance of the trainer to reuse its feature preparation method
        self.trainer = LSTMForexTrainer(db_file=self.db_file)

    def get_db_connection(self):
        """Get database connection to the SQLite file"""
        try:
            return sqlite3.connect(self.db_file)
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None

    def get_active_model_path(self, pair, timeframe):
        """Get the path of the active model for a given pair and timeframe."""
        conn = self.get_db_connection()
        if not conn:
            return None, None

        try:
            cursor = conn.cursor()
            query = """
                SELECT model_path FROM lstm_models
                WHERE pair = ? AND timeframe = ? AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (pair, timeframe))
            result = cursor.fetchone()
            if result:
                model_path = result[0]
                # The path in DB is relative to the project root, e.g., 'models/EURUSD_D1_lstm.h5'
                scaler_path = model_path.replace(self.models_dir, self.scalers_dir).replace('.h5', '_scaler.pkl')
                return model_path, scaler_path
            else:
                return None, None
        except Exception as e:
            print(f"❌ Error fetching active model path: {e}")
            return None, None
        finally:
            if conn:
                conn.close()

    def get_latest_data(self, pair, timeframe):
        """Fetch the latest data required for a single prediction."""
        # We need sequence_length + enough data to calculate technical indicators (e.g., 50 for MA50)
        limit = self.sequence_length + 100
        conn = self.get_db_connection()
        if not conn:
            return None

        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM forex_data
                WHERE pair = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(pair, timeframe, limit))
            if df.empty or len(df) < self.sequence_length:
                print(f"❌ Not enough data for {pair} {timeframe} to make a prediction.")
                return None
            
            # Sort by timestamp ascending for processing
            return df.sort_values('timestamp').reset_index(drop=True)

        except Exception as e:
            print(f"❌ Error fetching latest data: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def predict(self, pair, timeframe):
        """Make a prediction for a given pair and timeframe."""
        print(f"🔮 Making prediction for {pair} {timeframe}...")

        # 1. Get model and scaler paths
        model_path, scaler_path = self.get_active_model_path(pair, timeframe)
        if not model_path or not os.path.exists(model_path):
            return {"error": f"No active model found for {pair} {timeframe}. Please train a model first."}
        if not scaler_path or not os.path.exists(scaler_path):
            return {"error": f"Scaler file not found for {pair} {timeframe}."}

        # 2. Load model and scaler
        try:
            model = load_model(model_path, custom_objects={'mse': 'mse'})
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            return {"error": f"Failed to load model or scaler: {e}"}

        # 3. Get latest data
        df = self.get_latest_data(pair, timeframe)
        if df is None:
            return {"error": "Not enough data to make a prediction."}
        
        current_price = df['close'].iloc[-1]

        # 4. Prepare features
        feature_data, feature_columns = self.trainer.prepare_features(df)
        if feature_data is None or len(feature_data) < self.sequence_length:
            return {"error": "Failed to prepare features or not enough data after feature prep."}

        # 5. Get the last sequence
        last_sequence_raw = feature_data.tail(self.sequence_length)
        
        # 6. Scale the sequence
        scaled_sequence = scaler.transform(last_sequence_raw)
        
        # 7. Reshape for prediction
        X_pred = np.reshape(scaled_sequence, (1, self.sequence_length, len(feature_columns)))

        # 8. Predict
        predicted_scaled_price = model.predict(X_pred)[0][0]

        # 9. Inverse transform the prediction
        # Create a dummy array with the same shape as the scaler expects
        dummy_array = np.zeros((1, len(feature_columns)))
        # Place the predicted value in the 'close' price column's position
        close_price_index = feature_columns.index('close')
        dummy_array[0, close_price_index] = predicted_scaled_price
        # Inverse transform
        predicted_price = scaler.inverse_transform(dummy_array)[0, close_price_index]

        return {
            "current_price": float(current_price),
            "predicted_price": float(predicted_price)
        }
