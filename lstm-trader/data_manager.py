#!/usr/bin/env python3
"""
🔄 Real Forex Data Manager
Collects historical data from Alpha Vantage API and stores in PostgreSQL
"""

import os
import sys
import time
import json
import requests
import psycopg2
from datetime import datetime, timedelta
import pandas as pd

class ForexDataManager:
    def __init__(self, db_config=None):
        """Initialize data manager with database configuration"""
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'aiagent1_forex_lstm',
            'user': 'aiagent1',
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }

        # Alpha Vantage API configuration
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
        self.base_url = 'https://www.alphavantage.co/query'

        # Supported pairs and timeframes
        self.forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
        self.timeframes = {
            'D1': {'function': 'FX_DAILY', 'interval': None},
            'H4': {'function': 'FX_INTRADAY', 'interval': '60min'},  # Closest to H4
            'H1': {'function': 'FX_INTRADAY', 'interval': '60min'},
            'M30': {'function': 'FX_INTRADAY', 'interval': '30min'},
            'M15': {'function': 'FX_INTRADAY', 'interval': '15min'}
        }

    def get_db_connection(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None

    def fetch_forex_data(self, pair, timeframe='D1', outputsize='full'):
        """Fetch forex data from Alpha Vantage API"""
        print(f"📊 Fetching {pair} data for {timeframe}...")

        # Get timeframe configuration
        tf_config = self.timeframes.get(timeframe, self.timeframes['D1'])

        # Build API parameters
        params = {
            'function': tf_config['function'],
            'from_symbol': pair[:3],
            'to_symbol': pair[3:],
            'apikey': self.api_key,
            'outputsize': outputsize
        }

        # Add interval for intraday data
        if tf_config['interval']:
            params['interval'] = tf_config['interval']

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                print(f"❌ API Error: {data['Error Message']}")
                return None

            if 'Note' in data:
                print(f"⚠️  API Limit: {data['Note']}")
                return None

            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break

            if not time_series_key:
                print(f"❌ No time series data found for {pair}")
                return None

            return data[time_series_key]

        except Exception as e:
            print(f"❌ Error fetching {pair} data: {e}")
            return None

    def store_forex_data(self, pair, timeframe, data):
        """Store forex data in PostgreSQL database"""
        if not data:
            return 0

        conn = self.get_db_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()

            # Prepare data for insertion
            records = []
            for timestamp, prices in data.items():
                # Parse timestamp
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S' if ' ' in timestamp else '%Y-%m-%d')

                # Extract OHLC data
                open_price = float(prices.get('1. open', 0))
                high_price = float(prices.get('2. high', 0))
                low_price = float(prices.get('3. low', 0))
                close_price = float(prices.get('4. close', 0))
                volume = int(prices.get('5. volume', 0)) if '5. volume' in prices else 0

                records.append((pair, timeframe, dt, open_price, high_price, low_price, close_price, volume))

            # Insert data with conflict resolution
            insert_query = """
                INSERT INTO forex_data (pair, timeframe, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pair, timeframe, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """

            cursor.executemany(insert_query, records)

            # Log collection results
            log_query = """
                INSERT INTO data_collection_logs (pair, timeframe, records_added, provider, status)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(log_query, (pair, timeframe, len(records), 'alpha_vantage', 'success'))

            conn.commit()
            print(f"✅ Stored {len(records)} records for {pair} {timeframe}")
            return len(records)

        except Exception as e:
            print(f"❌ Error storing data: {e}")
            conn.rollback()
            return 0

        finally:
            conn.close()

    def collect_historical_data(self, pairs=None, timeframes=None, delay=12):
        """Collect historical data for specified pairs and timeframes"""
        pairs = pairs or self.forex_pairs
        timeframes = timeframes or ['D1', 'H1']  # Start with daily and hourly

        print(f"🚀 Starting data collection for {len(pairs)} pairs and {len(timeframes)} timeframes")
        total_records = 0

        for pair in pairs:
            for timeframe in timeframes:
                try:
                    # Fetch data from API
                    data = self.fetch_forex_data(pair, timeframe)

                    if data:
                        # Store in database
                        records_added = self.store_forex_data(pair, timeframe, data)
                        total_records += records_added

                    # Respect API rate limits
                    print(f"⏳ Waiting {delay} seconds...")
                    time.sleep(delay)

                except Exception as e:
                    print(f"❌ Error processing {pair} {timeframe}: {e}")
                    continue

        print(f"🎉 Data collection completed! Total records: {total_records}")
        return total_records

    def get_data_summary(self):
        """Get summary of collected data"""
        conn = self.get_db_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor()

            # Get data counts by pair and timeframe
            query = """
                SELECT pair, timeframe, COUNT(*) as records,
                       MIN(timestamp) as earliest,
                       MAX(timestamp) as latest
                FROM forex_data
                GROUP BY pair, timeframe
                ORDER BY pair, timeframe
            """

            cursor.execute(query)
            results = cursor.fetchall()

            print("\n📊 Data Summary:")
            print("-" * 70)
            print(f"{'Pair':<8} {'Timeframe':<10} {'Records':<10} {'From':<12} {'To':<12}")
            print("-" * 70)

            total_records = 0
            for row in results:
                pair, timeframe, count, earliest, latest = row
                total_records += count
                print(f"{pair:<8} {timeframe:<10} {count:<10} {earliest.strftime('%Y-%m-%d'):<12} {latest.strftime('%Y-%m-%d'):<12}")

            print("-" * 70)
            print(f"Total records: {total_records}")

            return results

        except Exception as e:
            print(f"❌ Error getting data summary: {e}")
            return None

        finally:
            conn.close()

def main():
    """Main function for command line usage"""
    print("🧠 LSTM Forex Data Manager")
    print("=" * 50)

    # Initialize data manager
    manager = ForexDataManager()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'collect':
            # Collect historical data
            pairs = sys.argv[2:] if len(sys.argv) > 2 else None
            manager.collect_historical_data(pairs=pairs)

        elif command == 'summary':
            # Show data summary
            manager.get_data_summary()

        else:
            print("Usage: python3 data_manager.py [collect|summary] [pairs...]")
            print("Example: python3 data_manager.py collect EURUSD GBPUSD")
    else:
        # Interactive mode - show summary
        manager.get_data_summary()

if __name__ == "__main__":
    main()