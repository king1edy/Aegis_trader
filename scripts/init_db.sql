-- =============================================================================
-- Trading Automation System - Database Initialization
-- =============================================================================
-- This script runs when the PostgreSQL container first starts.
-- It sets up TimescaleDB and creates the initial schema.
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to calculate win rate
CREATE OR REPLACE FUNCTION calculate_win_rate(wins INTEGER, total INTEGER)
RETURNS NUMERIC AS $$
BEGIN
    IF total = 0 THEN
        RETURN 0;
    END IF;
    RETURN ROUND((wins::NUMERIC / total::NUMERIC) * 100, 2);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate profit factor
CREATE OR REPLACE FUNCTION calculate_profit_factor(gross_profit NUMERIC, gross_loss NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    IF gross_loss = 0 OR gross_loss IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN ROUND(ABS(gross_profit / gross_loss), 2);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- Note: These views will be created after the tables exist.
-- The application's Alembic migrations handle the full schema.
-- This script just ensures the extensions are ready.

-- =============================================================================
-- Performance Indexes (additional to SQLAlchemy models)
-- =============================================================================

-- These will be created by Alembic, but we document them here for reference:
-- CREATE INDEX IF NOT EXISTS ix_trades_performance ON trades (strategy_name, outcome, entry_time);
-- CREATE INDEX IF NOT EXISTS ix_trades_daily_summary ON trades (DATE(entry_time), status);

-- =============================================================================
-- Grants (if using separate application user)
-- =============================================================================

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO trading_user;

-- Grant all privileges on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;

-- Grant all privileges on all sequences
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;

-- Default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;

-- =============================================================================
-- Notification Channel for Real-time Updates
-- =============================================================================

-- Create notification function for trade updates
CREATE OR REPLACE FUNCTION notify_trade_update()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'trade_updates',
        json_build_object(
            'operation', TG_OP,
            'trade_id', NEW.id,
            'ticket', NEW.ticket,
            'status', NEW.status,
            'symbol', NEW.symbol
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Note: The trigger will be created by Alembic after the trades table exists:
-- CREATE TRIGGER trade_update_trigger
-- AFTER INSERT OR UPDATE ON trades
-- FOR EACH ROW EXECUTE FUNCTION notify_trade_update();

RAISE NOTICE 'Database initialization complete';
