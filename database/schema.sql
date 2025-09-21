-- 🧠 LSTM Forex Database Schema
-- PostgreSQL tables for real forex prediction system

-- Історичні дані валютних пар
CREATE TABLE forex_data (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,  -- D1, H4, H1, M30, M15
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(10,5) NOT NULL,
    high DECIMAL(10,5) NOT NULL,
    low DECIMAL(10,5) NOT NULL,
    close DECIMAL(10,5) NOT NULL,
    volume BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(pair, timeframe, timestamp)
);

-- Навчені LSTM моделі
CREATE TABLE lstm_models (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    model_path TEXT NOT NULL,
    accuracy DECIMAL(5,2),
    loss DECIMAL(10,6),
    epochs INTEGER,
    training_samples INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(pair, timeframe, is_active) DEFERRABLE INITIALLY DEFERRED
);

-- Прогнози системи
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    model_id INTEGER REFERENCES lstm_models(id),
    current_price DECIMAL(10,5) NOT NULL,
    predicted_price DECIMAL(10,5) NOT NULL,
    price_change DECIMAL(10,5),
    price_change_percent DECIMAL(5,2),
    action VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
    confidence INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- API ключі та налаштування
CREATE TABLE api_settings (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL, -- alpha_vantage, yahoo, etc
    api_key TEXT,
    daily_limit INTEGER DEFAULT 25,
    used_today INTEGER DEFAULT 0,
    last_reset DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Логи збору даних
CREATE TABLE data_collection_logs (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    records_added INTEGER DEFAULT 0,
    provider VARCHAR(50),
    status VARCHAR(20), -- success, error, partial
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Індекси для швидкого пошуку
CREATE INDEX idx_forex_data_pair_timeframe ON forex_data(pair, timeframe);
CREATE INDEX idx_forex_data_timestamp ON forex_data(timestamp DESC);
CREATE INDEX idx_predictions_pair ON predictions(pair);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);

-- Початкові дані
INSERT INTO api_settings (provider, daily_limit) VALUES
('alpha_vantage', 25),
('yahoo_finance', 2000);

-- Коментарі
COMMENT ON TABLE forex_data IS 'Історичні дані валютних пар з різних таймфреймів';
COMMENT ON TABLE lstm_models IS 'Навчені LSTM моделі для прогнозування';
COMMENT ON TABLE predictions IS 'Прогнози створені системою';
COMMENT ON TABLE api_settings IS 'Налаштування API провайдерів';