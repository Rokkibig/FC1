-- 🧠 LSTM Forex Database Schema (SQLite Version)

-- Історичні дані валютних пар
CREATE TABLE IF NOT EXISTS forex_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL,  -- D1, H4, H1, M30, M15
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pair, timeframe, timestamp)
);

-- Навчені LSTM моделі
CREATE TABLE IF NOT EXISTS lstm_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_path TEXT NOT NULL,
    accuracy REAL,
    loss REAL,
    epochs INTEGER,
    training_samples INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Прогнози системи
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    model_id INTEGER REFERENCES lstm_models(id),
    current_price REAL NOT NULL,
    predicted_price REAL NOT NULL,
    price_change REAL,
    price_change_percent REAL,
    action TEXT NOT NULL, -- BUY, SELL, HOLD
    confidence INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- API ключі та налаштування
CREATE TABLE IF NOT EXISTS api_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    api_key TEXT,
    daily_limit INTEGER DEFAULT 25,
    used_today INTEGER DEFAULT 0,
    last_reset TEXT DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Логи збору даних
CREATE TABLE IF NOT EXISTS data_collection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    records_added INTEGER DEFAULT 0,
    provider TEXT,
    status TEXT, -- success, error, partial
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Індекси для швидкого пошуку
CREATE INDEX IF NOT EXISTS idx_forex_data_pair_timeframe ON forex_data(pair, timeframe);
CREATE INDEX IF NOT EXISTS idx_forex_data_timestamp ON forex_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_pair ON predictions(pair);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);

-- Початкові дані
INSERT OR IGNORE INTO api_settings (provider, daily_limit) VALUES
('alpha_vantage', 25),
('yahoo_finance', 2000);
