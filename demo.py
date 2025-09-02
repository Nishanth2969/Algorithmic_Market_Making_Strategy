#!/usr/bin/env python3
"""
Demo script for Algorithmic Market Making MVP
Runs a complete backtest and displays results
"""

import sys
import time
import argparse
from pathlib import Path

# Add the mm package to the path
sys.path.append(str(Path(__file__).parent))

from mm.backtest import run_backtest, run_parameter_sweep
from mm.metrics import MetricsCalculator
import yaml


def run_basic_demo():
    """Run basic demo backtest"""
    print("Algorithmic Market Making MVP Demo")
    print("=" * 50)
    
    config_path = Path(__file__).parent / "configs" / "default.yml"
    
    print(f"Loading configuration from: {config_path}")
    print("Starting backtest...")
    print()
    
    start_time = time.time()
    
    try:
        # Run the backtest
        results = run_backtest(str(config_path), verbose=True)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.1f} seconds")
        
        # Display detailed results
        print_detailed_results(results)
        
        # Validate against targets

        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


def run_scenario_demo(scenario: str):
    """Run scenario-specific demo"""
    print(f"Running scenario: {scenario}")
    print("=" * 50)
    
    config_path = Path(__file__).parent / "configs" / "default.yml"
    
    # Load base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config based on scenario
    if scenario == "vol_spike":
        config['market_data']['volatility'] = 0.05  # 5% volatility
        print("Simulating high volatility market conditions")
        
    elif scenario == "thin_book":
        config['market_data']['volatility'] = 0.01
        config['arrival_model']['A'] = 0.5  # Lower liquidity
        print("Simulating thin market conditions")
        
    elif scenario == "coint_break":
        config['stat_arb']['enabled'] = False
        print("Simulating cointegration breakdown")
        
    elif scenario == "risk_test":
        config['risk']['dd_stop_pct'] = 0.01  # 1% drawdown limit
        config['risk']['q_max'] = 100  # Lower position limit
        print("Testing aggressive risk limits")
        
    else:
        print(f"Unknown scenario: {scenario}")
        return
    
    # Save modified config
    temp_config_path = Path(__file__).parent / f"temp_config_{scenario}.yml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        results = run_backtest(str(temp_config_path), verbose=True)
        print_detailed_results(results)
        
        # Clean up temp file
        temp_config_path.unlink()
        
    except Exception as e:
        print(f"Scenario demo failed: {e}")
        # Clean up temp file
        if temp_config_path.exists():
            temp_config_path.unlink()


def run_parameter_sweep_demo():
    """Run parameter optimization demo"""
    print("🔧 Parameter Optimization Demo")
    print("=" * 50)
    
    config_path = Path(__file__).parent / "configs" / "default.yml"
    
    # Define parameter grid
    param_grid = {
        'risk.gamma': [0.001, 0.005, 0.01],
        'arrival_model.k': [0.8, 1.2, 1.6],
        'risk.inv_penalty_phi': [0.01, 0.02, 0.03]
    }
    
    print("Testing parameter combinations:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()
    
    try:
        results_list = run_parameter_sweep(str(config_path), param_grid, verbose=False)
        
        print(f"✅ Completed {len(results_list)} parameter combinations")
        
        # Find best result
        best_result = max(results_list, key=lambda r: r.sharpe_ratio)
        
        print(f"\n🏆 Best Result:")
        print(f"  Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
        print(f"  Total Return: {best_result.total_return:.1%}")
        print(f"  Win Rate: {best_result.win_rate:.1%}")
        print(f"  Max Drawdown: {best_result.max_drawdown:.1%}")
        
        # Show parameter sweep summary
        print("\n📊 Parameter Sweep Summary:")
        print("-" * 40)
        
        sharpe_ratios = [r.sharpe_ratio for r in results_list]
        returns = [r.total_return for r in results_list]
        
        print(f"Sharpe Ratio Range: {min(sharpe_ratios):.2f} - {max(sharpe_ratios):.2f}")
        print(f"Return Range: {min(returns):.1%} - {max(returns):.1%}")
        
    except Exception as e:
        print(f"Parameter sweep failed: {e}")


def print_detailed_results(results):
    """Print detailed backtest results"""
    print("\nDETAILED RESULTS")
    print("=" * 50)
    
    # Key Performance Metrics
    print("Key Performance Metrics:")
    print(f"  Total Return:       {results.total_return:.1%}")
    print(f"  Sharpe Ratio:       {results.sharpe_ratio:.2f}")
    print(f"  Win Rate:           {results.win_rate:.1%}")
    print(f"  Max Drawdown:       {results.max_drawdown:.1%}")
    print(f"  Number of Trades:   {results.num_trades:,}")
    
    # Portfolio Metrics
    final_pnl = results.state.total_pnl
    final_value = results.state.portfolio_value
    initial_capital = 100000  # From config
    
    print(f"\nPortfolio Summary:")
    print(f"  Initial Capital:    ${initial_capital:,.0f}")
    print(f"  Final Value:        ${final_value:,.0f}")
    print(f"  Total P&L:          ${final_pnl:,.0f}")
    print(f"  Final Position:     {results.state.position:.1f}")
    
    # Strategy Performance
    metrics = results.metrics
    if 'market_making' in metrics:
        mm_metrics = metrics['market_making']
        print(f"\nMarket Making Metrics:")
        print(f"  Fill Rate:          {mm_metrics.get('fill_rate', 0):.1%}")
        print(f"  Quote Count:        {mm_metrics.get('quote_count', 0):,}")
        print(f"  Inventory Turnover: {mm_metrics.get('inventory_turnover', 0):.1f}x")
    
    # Risk Metrics
    if 'risk' in metrics:
        risk_metrics = metrics['risk']
        print(f"\nRisk Analysis:")
        print(f"  Max Position:       {risk_metrics.get('max_position', 0):.1f}")
        print(f"  Average Position:   {risk_metrics.get('avg_position', 0):.1f}")
        print(f"  Position Volatility: {risk_metrics.get('position_volatility', 0):.1f}")
    
    # Attribution Analysis
    if 'attribution' in metrics:
        attr = metrics['attribution']
        print(f"\nPerformance Attribution:")
        print(f"  P&L from Spread:    ${attr.get('pnl_from_spread', 0):,.0f}")
        print(f"  P&L from Stat-Arb:  ${attr.get('pnl_from_stat_arb', 0):,.0f}")
        print(f"  P&L from Inventory: ${attr.get('pnl_from_inventory', 0):,.0f}")





def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Algorithmic Market Making Demo')
    parser.add_argument('--scenario', choices=['vol_spike', 'thin_book', 'coint_break', 'risk_test'],
                       help='Run specific scenario test')
    parser.add_argument('--param-sweep', action='store_true',
                       help='Run parameter optimization demo')
    
    args = parser.parse_args()
    
    try:
        if args.param_sweep:
            run_parameter_sweep_demo()
        elif args.scenario:
            run_scenario_demo(args.scenario)
        else:
            run_basic_demo()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
