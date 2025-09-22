#!/usr/bin/env python3
"""
🔄 Real Forex Data Manager
Collects historical data from Yahoo Finance and stores in an SQLite database.
"""

import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env файлу (хоча для yfinance вони не потрібні)
load_dotenv()

class ForexDataManager:
    def __init__(self, db_file="forex_data.db"):
        """Initialize data manager with SQLite database file"""
        self.db_file = db_file
        self.get_db_connection() # Initialize DB if not exists

        # Supported pairs and timeframes
        self.forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
        # yfinance uses different interval codes
        self.timeframes = {
            'D1': '1d',
            'H4': '4h',
            'H1': '1h',
            'M30': '30m',
            'M15': '15m'
        }

    def get_db_connection(self):
        """Create database connection to the SQLite file."""
        try:
            conn = sqlite3.connect(self.db_file)
            return conn
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None

    def fetch_forex_data(self, pair, timeframe_key='D1', period="5y"):
        """Fetch forex data from Yahoo Finance API"""
        print(f"📊 Fetching {pair} data for timeframe {timeframe_key} using yfinance...")
        
        interval = self.timeframes.get(timeframe_key, '1d')
        ticker = f"{pair}=X"

        # yfinance has a 730-day limit for intraday data
        period_to_fetch = period
        if interval not in ['1d', '1wk', '1mo']:
            print("⏳ Adjusting period to 2 years for intraday data...")
            period_to_fetch = "729d"

        try:
            data = yf.download(ticker, period=period_to_fetch, interval=interval, auto_adjust=True, progress=False)

            if data.empty:
                print(f"❌ No data found for {ticker} with interval {interval}")
                return None

            # Rename columns to match our schema (lowercase)
            data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            return data

        except Exception as e:
            print(f"❌ Error fetching {pair} data: {e}")
            return None

    def store_forex_data(self, pair, timeframe_key, data):
        """Store forex data from a DataFrame in SQLite database"""
        if data is None or data.empty:
            return 0

        conn = self.get_db_connection()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()

            # Prepare data for insertion
            records = []
            for timestamp, row in data.iterrows():
                dt_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                # Explicitly cast types to prevent binding errors
                records.append((
                    pair, 
                    timeframe_key, 
                    dt_str, 
                    float(row['open']), 
                    float(row['high']), 
                    float(row['low']), 
                    float(row['close']), 
                    int(row['volume'])
                ))

            # Insert data with conflict resolution (UPSERT)
            insert_query = """
                INSERT INTO forex_data (pair, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (pair, timeframe, timestamp) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume
            """

            cursor.executemany(insert_query, records)

            # Log collection results
            log_query = """
                INSERT INTO data_collection_logs (pair, timeframe, records_added, provider, status)
                VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(log_query, (pair, timeframe_key, len(records), 'yfinance', 'success'))

            conn.commit()
            print(f"✅ Stored {len(records)} records for {pair} {timeframe_key}")
            return len(records)

        except Exception as e:
            print(f"❌ Error storing data: {e}")
            conn.rollback()
            return 0

        finally:
            conn.close()

    def collect_historical_data(self, pairs=None, timeframes=None):
        """Collect historical data for specified pairs and timeframes"""
        pairs = pairs or self.forex_pairs
        timeframes = timeframes or ['D1', 'H1']  # Default timeframes to collect

        print(f"🚀 Starting data collection for {len(pairs)} pairs and {len(timeframes)} timeframes")
        total_records = 0

        for pair in pairs:
            for tf_key in timeframes:
                try:
                    # Fetch data from API
                    data = self.fetch_forex_data(pair, tf_key)

                    if data is not None:
                        # Store in database
                        records_added = self.store_forex_data(pair, tf_key, data)
                        total_records += records_added
                    
                    # Small delay to be polite to the API server
                    time.sleep(1)

                except Exception as e:
                    print(f"❌ Error processing {pair} {tf_key}: {e}")
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
                # Timestamps from SQLite are strings, print them directly
                print(f"{pair:<8} {timeframe:<10} {count:<10} {earliest.split(' ')[0]:<12} {latest.split(' ')[0]:<12}")

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