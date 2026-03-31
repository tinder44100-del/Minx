#!/usr/bin/env python3
# =========================================
# 🚀 XAUUSD Trading System - Main Entry Point
# =========================================
"""
XAUUSD 5-Minute Trading System
Ready for Production with Cent Account

Features:
- Real-time data fetching from Binance
- Advanced feature engineering for gold
- ML-based signal generation (LightGBM)
- Walk-forward validation
- Risk management
- Full backtesting framework
"""

import os
import sys
import yaml
import logging
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import XAUUSDDataFetcher, generate_realistic_market
from src.feature_engineering import (
    XAUUSDFeatureEngine, add_state_machine, add_target, get_feature_list
)
from src.model import XAUUSDModel, evaluate_model_performance
from src.trading_engine import TradingEngine


def setup_logging(log_dir: str = 'logs'):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_dir}/trading_system_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_backtest(config: dict, use_synthetic: bool = True, days: int = 30) -> dict:
    """
    Run complete backtest of the trading system.

    Args:
        config: Configuration dictionary
        use_synthetic: Use synthetic data if True
        days: Days of data to fetch if using real data

    Returns:
        Dictionary with backtest results
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("XAUUSD TRADING SYSTEM - BACKTEST")
    logger.info("=" * 70)

    # =========================================
    # 1. Load Data
    # =========================================
    logger.info("\n📊 Step 1: Loading Market Data...")

    if use_synthetic:
        logger.info("Using synthetic market data for testing")
        df = generate_realistic_market(n=5000, config=config)
    else:
        logger.info("Fetching real XAUUSD data from Binance")
        fetcher = XAUUSDDataFetcher(config)
        try:
            df = fetcher.get_recent_data(days=days)
        except Exception as e:
            logger.warning(f"Failed to fetch data: {e}, using synthetic")
            df = generate_realistic_market(n=5000, config=config)

    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # =========================================
    # 2. Feature Engineering
    # =========================================
    logger.info("\n📈 Step 2: Feature Engineering...")

    feature_engine = XAUUSDFeatureEngine(config)
    df = feature_engine.add_all_features(df)

    logger.info("Adding state machine features...")
    df = add_state_machine(df, config)

    logger.info("Adding target labels...")
    df = add_target(
        df,
        look_forward=1,
        tp_pips=config.get('RISK', {}).get('take_profit_pips', 100),
        sl_pips=config.get('RISK', {}).get('stop_loss_pips', 50)
    )

    features = get_feature_list(config)
    features = [f for f in features if f in df.columns]

    logger.info(f"Total features: {len(features)}")
    logger.info(f"Sample features: {features[:10]}")

    # =========================================
    # 3. Train Model
    # =========================================
    logger.info("\n🤖 Step 3: Training ML Model...")

    model = XAUUSDModel(config)

    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]

    logger.info(f"Training set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")

    metrics = model.train(
        train_df[features],
        train_df['target'],
        val_df[features],
        val_df['target']
    )

    logger.info(f"Training Accuracy: {metrics['train']['accuracy']:.3f}")
    logger.info(f"Validation Accuracy: {metrics['validation']['accuracy']:.3f}")
    logger.info(f"Validation F1: {metrics['validation']['f1']:.3f}")

    # Top features
    top_features = model.get_top_features(10)
    logger.info("\nTop 10 Features:")
    for _, row in top_features.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.0f}")

    # =========================================
    # 4. Walk-Forward Backtest
    # =========================================
    logger.info("\n🎯 Step 4: Walk-Forward Backtest...")

    wf_config = config.get('WALK_FORWARD', {})
    train_window = wf_config.get('train_window', 1500)
    test_window = wf_config.get('test_window', 500)
    step = wf_config.get('step', 250)

    all_results = []
    all_trades = []

    total_windows = (len(df) - train_window - test_window) // step + 1
    logger.info(f"Total windows to process: {total_windows}")

    for i, start in enumerate(range(0, len(df) - train_window - test_window, step)):
        train_end = start + train_window
        test_end = train_end + test_window

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end].copy()

        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"\nProcessing window {i + 1}/{total_windows}...")

        # Retrain model
        model.model = None  # Reset
        model.train(train_df[features], train_df['target'])

        # Predict
        test_df['probability'] = model.predict_proba(test_df[features])[:, 1]

        # Simulate trading
        trading_engine = TradingEngine(config)

        for idx, row in test_df.iterrows():
            trading_engine.execute_bar(row)

        # Collect results
        trades_df = trading_engine.get_trades_df()
        summary = trading_engine.get_summary()

        window_result = {
            'window': i + 1,
            'start': start,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'accuracy': model.metrics.get('validation', {}).get('accuracy', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'total_pnl': summary.get('total_pnl', 0),
            'profit_factor': summary.get('profit_factor', 0),
            'max_drawdown': summary.get('max_drawdown', 0)
        }

        all_results.append(window_result)

        if len(trades_df) > 0:
            trades_df['window'] = i + 1
            all_trades.append(trades_df)

    # =========================================
    # 5. Results Summary
    # =========================================
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("=" * 70)

    results_df = pd.DataFrame(all_results)

    logger.info(f"Total Windows: {len(results_df)}")
    logger.info(f"Total Trades: {results_df['total_trades'].sum()}")
    logger.info(f"Average Win Rate: {results_df['win_rate'].mean():.1%}")
    logger.info(f"Average Accuracy: {results_df['accuracy'].mean():.3f}")
    logger.info(f"Total PnL: ${results_df['total_pnl'].sum():.2f}")
    logger.info(f"Max Drawdown: {results_df['max_drawdown'].max():.1%}")

    # Trading statistics
    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        winning = all_trades_df[all_trades_df['pnl'] > 0]
        losing = all_trades_df[all_trades_df['pnl'] <= 0]

        logger.info(f"\nTrade Statistics:")
        logger.info(f"  Winning Trades: {len(winning)}")
        logger.info(f"  Losing Trades: {len(losing)}")
        logger.info(f"  Win Rate: {len(winning) / len(all_trades_df) * 100:.1f}%")
        logger.info(f"  Average Win: ${winning['pnl'].mean():.2f}" if len(winning) > 0 else "  Average Win: N/A")
        logger.info(f"  Average Loss: ${losing['pnl'].mean():.2f}" if len(losing) > 0 else "  Average Loss: N/A")
        logger.info(f"  Largest Win: ${all_trades_df['pnl'].max():.2f}")
        logger.info(f"  Largest Loss: ${all_trades_df['pnl'].min():.2f}")

    return {
        'results': results_df,
        'trades': pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        'config': config,
        'features': features
    }


def export_results(results: dict, output_dir: str = 'results') -> None:
    """Export backtest results to files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save results
    results['results'].to_csv(f'{output_dir}/backtest_results_{timestamp}.csv', index=False)

    # Save trades
    if len(results['trades']) > 0:
        results['trades'].to_csv(f'{output_dir}/trades_{timestamp}.csv', index=False)

    # Save config
    with open(f'{output_dir}/config_{timestamp}.yaml', 'w') as f:
        yaml.dump(results['config'], f)

    logging.getLogger(__name__).info(f"\n✅ Results exported to {output_dir}/")


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Print configuration
    logger.info("\n📋 Configuration:")
    logger.info(f"  Symbol: {config['SYMBOL']}")
    logger.info(f"  Timeframe: {config['TIMEFRAME']}")
    logger.info(f"  Initial Balance: ${config['ACCOUNT']['balance'] / 100:.2f}")
    logger.info(f"  Risk per Trade: {config['RISK']['max_risk_per_trade']:.1%}")
    logger.info(f"  Stop Loss: {config['RISK']['stop_loss_pips']} pips")
    logger.info(f"  Take Profit: {config['RISK']['take_profit_pips']} pips")

    # Run backtest
    try:
        results = run_backtest(config, use_synthetic=True)

        # Export results
        export_results(results)

        logger.info("\n✅ Backtest completed successfully!")

    except Exception as e:
        logging.getLogger(__name__).error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
