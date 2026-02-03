"""
Setup Verification Script
=========================
Run this to verify the trading system is configured correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def verify_setup():
    """Verify all components are properly configured."""
    print("=" * 60)
    print("AEGIS AUTOMATED TRADING SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Check imports
    print("\n[1/5] Checking imports...")
    try:
        from src.core.config import settings
        print("  ✓ Core configuration loaded")
    except Exception as e:
        errors.append(f"Failed to import config: {e}")
        print(f"  ✗ Configuration error: {e}")
    
    try:
        from src.core.logging_config import get_logger
        logger = get_logger("test")
        print("  ✓ Logging configured")
    except Exception as e:
        errors.append(f"Failed to import logging: {e}")
        print(f"  ✗ Logging error: {e}")
    
    try:
        from src.database.models import Trade, AccountSnapshot
        print("  ✓ Database models loaded")
    except Exception as e:
        errors.append(f"Failed to import models: {e}")
        print(f"  ✗ Models error: {e}")
    
    try:
        from src.execution.mt5_connector import DemoConnector, MT5Connector
        print("  ✓ Execution module loaded")
    except Exception as e:
        errors.append(f"Failed to import execution: {e}")
        print(f"  ✗ Execution error: {e}")
    
    # 2. Validate configuration
    print("\n[2/5] Validating configuration...")
    try:
        from src.core.config import settings
        config_warnings = settings.validate_trading_config()
        for w in config_warnings:
            warnings.append(w)
            print(f"  ⚠ {w}")
        if not config_warnings:
            print("  ✓ Configuration valid")
    except Exception as e:
        errors.append(f"Configuration validation failed: {e}")
    
    # 3. Test demo connector
    print("\n[3/5] Testing demo connector...")
    try:
        from src.execution import DemoConnector
        connector = DemoConnector(initial_balance=10000)
        await connector.connect()
        
        account = await connector.get_account_info()
        print(f"  ✓ Demo account: ${account.balance}")
        
        symbol_info = await connector.get_symbol_info("XAUUSD")
        print(f"  ✓ Symbol info: {symbol_info.name}, digits={symbol_info.digits}")
        
        await connector.disconnect()
        print("  ✓ Demo connector working")
    except Exception as e:
        errors.append(f"Demo connector test failed: {e}")
        print(f"  ✗ Demo connector error: {e}")
    
    # 4. Check environment
    print("\n[4/5] Checking environment...")
    try:
        from src.core.config import settings
        print(f"  Environment: {settings.app_env.value}")
        print(f"  Debug mode: {settings.debug}")
        print(f"  Log level: {settings.log_level}")
        print(f"  Max risk per trade: {settings.max_risk_per_trade * 100}%")
        print(f"  Max daily risk: {settings.max_daily_risk * 100}%")
        print(f"  Manual override: {'ENABLED ⚠' if settings.enable_manual_override else 'DISABLED ✓'}")
    except Exception as e:
        warnings.append(f"Environment check issue: {e}")
    
    # 5. Database connection test (skip if no DB running)
    print("\n[5/5] Database configuration...")
    try:
        from src.core.config import settings
        print(f"  DB URL: {settings.db_url[:50]}...")
        print(f"  Redis URL: {settings.redis_connection_url}")
        print("  ✓ Database configuration present")
        print("  ℹ Run 'docker-compose up -d postgres redis' to start databases")
    except Exception as e:
        warnings.append(f"Database config issue: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e}")
    
    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    
    if not errors:
        print("\n✅ All core components verified successfully!")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and configure your settings")
        print("  2. Run: docker-compose up -d")
        print("  3. Check logs: docker-compose logs -f trading_app")
    else:
        print("\n❌ Some components failed verification. Please fix errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(verify_setup())
    sys.exit(exit_code)
