#!/usr/bin/env python3
"""
🧠 Real LSTM Model Trainer
Advanced neural network training for forex prediction
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import pickle
import json

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("✅ TensorFlow successfully imported")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("💡 Install with: pip install tensorflow scikit-learn")

class LSTMForexTrainer:
    def __init__(self, db_config=None):
        """Initialize LSTM trainer"""
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'aiagent1_forex_lstm',
            'user': 'aiagent1',
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }

        # Model parameters
        self.sequence_length = 60  # 60 time steps for prediction
        self.prediction_steps = 1  # Predict 1 step ahead
        self.features = ['open', 'high', 'low', 'close', 'volume']

        # Model architecture
        self.lstm_units = [64, 32, 16]
        self.dropout_rate = 0.2
        self.batch_size = 32
        self.epochs = 100

        # Paths
        self.models_dir = 'models'
        self.scalers_dir = 'scalers'

        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)

    def get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None

    def fetch_training_data(self, pair, timeframe, limit=10000):
        """Fetch training data from database"""
        conn = self.get_db_connection()
        if not conn:
            return None

        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM forex_data
                WHERE pair = %s AND timeframe = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """

            df = pd.read_sql_query(query, conn, params=(pair, timeframe, limit))

            if df.empty:
                print(f"❌ No data found for {pair} {timeframe}")
                return None

            # Sort by timestamp ascending for training
            df = df.sort_values('timestamp').reset_index(drop=True)

            print(f"📊 Loaded {len(df)} records for {pair} {timeframe}")
            print(f"📅 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            print(f"❌ Error fetching training data: {e}")
            return None
        finally:
            conn.close()

    def prepare_features(self, df):
        """Prepare features for LSTM training"""
        try:
            # Create technical indicators
            df = df.copy()

            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']

            # Moving averages
            df['ma_5'] = df['close'].rolling(window=5).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()

            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()

            # RSI (simplified)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Select final features
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'price_change', 'high_low_ratio', 'close_open_ratio',
                'ma_5', 'ma_20', 'ma_50', 'volatility', 'rsi',
                'volume_ratio'
            ]

            # Remove rows with NaN values
            df = df.dropna()

            print(f"✅ Prepared {len(df)} samples with {len(feature_columns)} features")
            return df[feature_columns], feature_columns

        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None, None

    def create_sequences(self, data, target_column='close'):
        """Create sequences for LSTM training"""
        try:
            X, y = [], []

            for i in range(self.sequence_length, len(data)):
                # Input sequence
                X.append(data[i-self.sequence_length:i])

                # Target (next close price)
                y.append(data[i][data.columns.get_loc(target_column)])

            return np.array(X), np.array(y)

        except Exception as e:
            print(f"❌ Error creating sequences: {e}")
            return None, None

    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        try:
            model = Sequential()

            # First LSTM layer
            model.add(LSTM(
                units=self.lstm_units[0],
                return_sequences=True,
                input_shape=input_shape
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

            # Second LSTM layer
            model.add(LSTM(
                units=self.lstm_units[1],
                return_sequences=True
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

            # Third LSTM layer
            model.add(LSTM(
                units=self.lstm_units[2],
                return_sequences=False
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

            # Dense layers
            model.add(Dense(units=32, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=1))  # Single output for price prediction

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

            print("✅ LSTM model architecture created")
            model.summary()

            return model

        except Exception as e:
            print(f"❌ Error building model: {e}")
            return None

    def train_model(self, pair, timeframe):
        """Train LSTM model for specific pair and timeframe"""
        print(f"🚀 Training LSTM model for {pair} {timeframe}")

        # Fetch training data
        df = self.fetch_training_data(pair, timeframe)
        if df is None:
            return None

        # Prepare features
        feature_data, feature_columns = self.prepare_features(df)
        if feature_data is None:
            return None

        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)

        # Create sequences
        X, y = self.create_sequences(pd.DataFrame(scaled_data, columns=feature_columns))
        if X is None:
            return None

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Build model
        model = self.build_model((X.shape[1], X.shape[2]))
        if model is None:
            return None

        # Training callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]

        # Train model
        print("🧠 Starting model training...")
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)

        print(f"📈 Training Loss: {train_loss[0]:.6f}")
        print(f"📉 Test Loss: {test_loss[0]:.6f}")

        # Save model and scaler
        model_filename = f"{pair}_{timeframe}_lstm.h5"
        scaler_filename = f"{pair}_{timeframe}_scaler.pkl"

        model_path = os.path.join(self.models_dir, model_filename)
        scaler_path = os.path.join(self.scalers_dir, scaler_filename)

        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save to database
        self.save_model_to_db(pair, timeframe, model_path, test_loss[0], len(X_train))

        print(f"✅ Model saved: {model_path}")
        return model_path

    def save_model_to_db(self, pair, timeframe, model_path, loss, training_samples):
        """Save model metadata to database"""
        conn = self.get_db_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()

            # Deactivate old models
            cursor.execute("""
                UPDATE lstm_models
                SET is_active = FALSE
                WHERE pair = %s AND timeframe = %s AND is_active = TRUE
            """, (pair, timeframe))

            # Insert new model
            cursor.execute("""
                INSERT INTO lstm_models (pair, timeframe, model_path, loss, training_samples)
                VALUES (%s, %s, %s, %s, %s)
            """, (pair, timeframe, model_path, loss, training_samples))

            conn.commit()
            print(f"✅ Model metadata saved to database")

        except Exception as e:
            print(f"❌ Error saving model to database: {e}")
            conn.rollback()
        finally:
            conn.close()

def main():
    """Main function for command line usage"""
    print("🧠 LSTM Forex Model Trainer")
    print("=" * 50)

    trainer = LSTMForexTrainer()

    if len(sys.argv) >= 3:
        pair = sys.argv[1].upper()
        timeframe = sys.argv[2].upper()

        print(f"🎯 Training model for {pair} {timeframe}")
        model_path = trainer.train_model(pair, timeframe)

        if model_path:
            print(f"🎉 Training completed successfully!")
        else:
            print(f"❌ Training failed!")
    else:
        print("Usage: python3 model_trainer.py PAIR TIMEFRAME")
        print("Example: python3 model_trainer.py EURUSD D1")

if __name__ == "__main__":
    main()