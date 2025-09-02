"""Event-driven backtesting framework."""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path

from .datafeed import create_datafeed, MarketEvent, QuoteEvent, TradeEvent
from .orderbook import OrderBook, Side
from .execution import ExecutionEngine, OrderRequest, OrderType
from .signals import SignalManager
from .quoting import QuotingEngine
from .risk import RiskManager
from .metrics import MetricsCalculator


@dataclass
class BacktestState:

    timestamp: float = 0.0
    step: int = 0
    total_steps: int = 0
    is_running: bool = False
    is_complete: bool = False
    
    # Market state
    current_price: float = 100.0
    current_spread: float = 0.01
    
    # Portfolio state
    cash: float = 100000.0
    position: float = 0.0
    portfolio_value: float = 100000.0
    total_pnl: float = 0.0
    
    # Trading stats
    num_trades: int = 0
    num_orders: int = 0
    num_cancels: int = 0


@dataclass
class BacktestResults:
    """Results of a completed backtest"""
    config: Dict[str, Any]
    state: BacktestState
    metrics: Dict[str, Any]
    trade_log: List[Dict[str, Any]]
    risk_log: List[Dict[str, Any]]
    signal_log: List[Dict[str, Any]]
    quote_log: List[Dict[str, Any]]
    
    # Performance summary
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config_path: str):
        """Initialize backtest engine with configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize state
        self.state = BacktestState()
        self.state.cash = self.config.get('backtest', {}).get('start_cash', 100000)
        self.state.portfolio_value = self.state.cash
        
        # Initialize components
        self._initialize_components()
        
        # Logging
        self.trade_log: List[Dict[str, Any]] = []
        self.risk_log: List[Dict[str, Any]] = []
        self.signal_log: List[Dict[str, Any]] = []
        self.quote_log: List[Dict[str, Any]] = []
        
        # Results
        self.results: Optional[BacktestResults] = None
        
    def _initialize_components(self):
        """Initialize all strategy components"""
        
        # Market data feed
        market_config = self.config.copy()
        market_config['mode'] = self.config.get('backtest', {}).get('mode', 'sim')
        market_config['max_events'] = self._calculate_max_events()
        
        # Setup symbols for stat-arb if enabled
        if self.config.get('stat_arb', {}).get('enabled', False):
            pairs = self.config.get('pairs', [])
            if pairs:
                symbols = [pairs[0]['symbol1'], pairs[0]['symbol2']]
            else:
                symbols = ['ASSET1', 'ASSET2']  # Default pair
        else:
            symbols = ['ASSET1']
        
        market_config['symbols'] = symbols
        self.datafeed = create_datafeed(market_config)
        
        # Order book
        self.orderbook = OrderBook(symbols[0], self.config.get('simulation', {}).get('tick_size', 0.01))
        
        # Execution engine
        exec_config = self.config.get('simulation', {})
        exec_config.update(self.config.get('risk', {}))
        self.execution = ExecutionEngine(exec_config, self.orderbook)
        
        # Signal manager
        signal_config = self.config.copy()
        self.signals = SignalManager(signal_config)
        
        # Add cointegration pairs if enabled
        if len(symbols) > 1:
            self.signals.add_cointegration_pair(symbols[0], symbols[1])
        
        # Quoting engine
        quoting_config = self.config.get('risk', {})
        quoting_config.update(self.config.get('arrival_model', {}))
        quoting_config.update(self.config.get('sizing', {}))
        quoting_config.update(self.config.get('stat_arb', {}))
        self.quoting = QuotingEngine(quoting_config)
        
        # Risk manager
        risk_config = self.config.get('risk', {})
        risk_config['start_cash'] = self.state.cash
        self.risk_manager = RiskManager(risk_config)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(self.config)
        
    def _calculate_max_events(self) -> int:
        """Calculate maximum events needed to reach trade target"""
        target_trades = self.config.get('backtest', {}).get('trade_target', 1500)
        days = self.config.get('backtest', {}).get('days', 5)
        
        # Estimate events needed (rough heuristic)
        events_per_trade = 50  # Roughly 50 market events per trade
        max_events = target_trades * events_per_trade
        
        # Cap by time horizon
        max_events_per_day = 10000  # Reasonable limit
        time_cap = days * max_events_per_day
        
        return min(max_events, time_cap)
    
    def run(self, verbose: bool = True) -> BacktestResults:
        """Run the backtest"""
        
        if verbose:
            print(f"Starting backtest with config:")
            print(f"  Mode: {self.config.get('backtest', {}).get('mode', 'sim')}")
            print(f"  Target trades: {self.config.get('backtest', {}).get('trade_target', 1500)}")
            print(f"  Days: {self.config.get('backtest', {}).get('days', 5)}")
            print(f"  Starting cash: ${self.state.cash:,.0f}")
        
        self.state.is_running = True
        self.state.is_complete = False
        
        try:
            # Main event loop
            for event in self.datafeed.stream():
                if not self._process_event(event):
                    break  # Stop condition met
                
                self.state.step += 1
                
                # Progress reporting
                if verbose and self.state.step % 1000 == 0:
                    self._print_progress()
                
                # Check completion conditions
                if self._should_stop():
                    break
            
            # Finalize backtest
            self._finalize_backtest()
            
            if verbose:
                print("\nBacktest completed!")
                self._print_final_summary()
            
        except KeyboardInterrupt:
            print("\nBacktest interrupted by user")
            self._finalize_backtest()
        except Exception as e:
            print(f"\nBacktest failed with error: {e}")
            raise
        
        finally:
            self.state.is_running = False
            self.state.is_complete = True
        
        return self.results
    
    def _process_event(self, event: MarketEvent) -> bool:
        """Process a single market event"""
        
        try:
            self.state.timestamp = event.timestamp
            
            # Update orderbook with market data
            fills = self.orderbook.update_from_market_data(event)
            
            # Process any fills
            if fills:
                execution_reports = self.execution.process_fills(fills, event.timestamp)
                for report in execution_reports:
                    self._log_trade(report)
            
            # Step execution engine (process pending orders)
            execution_reports = self.execution.step(event.timestamp)
            for report in execution_reports:
                self._log_trade(report)
            
            # Update market data for signals
            market_data = self._extract_market_data(event)
            
            # Add pairs data for stat-arb if available
            if hasattr(event, 'symbol') and len(self.datafeed.feeds if hasattr(self.datafeed, 'feeds') else {}) > 1:
                # For multi-asset feeds, we'd need to track prices by symbol
                # For now, use simple correlation for demo
                price = market_data.get('mid_price', 100.0)
                correlated_price = price * 0.95 + np.random.normal(0, 0.001)  # Slightly correlated
                market_data['pairs_data'] = {
                    'ASSET1_ASSET2': (price, correlated_price)
                }
            
            # Update signals
            signals = self.signals.update_signals(market_data)
            self._log_signals(signals)
            
            # Update portfolio state
            portfolio_state = self._get_portfolio_state()
            
            # Update risk metrics
            risk_metrics = self.risk_manager.update_risk_metrics(portfolio_state, event.timestamp)
            self._log_risk(risk_metrics)
            
            # Check if we should quote
            risk_check = {
                'kill_switch_active': risk_metrics.kill_switch_active,
                'position_limit_breach': risk_metrics.position_limit_breach
            }
            
            should_quote = self.quoting.should_quote(market_data, risk_check)
            
            if should_quote and not risk_metrics.kill_switch_active:
                # Generate quote
                position = self.execution.get_position(self.orderbook.symbol).quantity
                time_to_close = self._get_time_to_close()
                
                quote = self.quoting.generate_quote(
                    market_data, position, signals, time_to_close
                )
                
                if quote:
                    self._log_quote(quote)
                    
                    # Create order requests
                    active_orders = self.execution.get_active_orders()
                    order_requests = self.quoting.create_order_requests(quote, active_orders)
                    
                    # Cancel existing orders if needed
                    if order_requests and active_orders:
                        for order in active_orders:
                            self.execution.cancel_order(order.order_id, event.timestamp)
                    
                    # Submit new orders
                    for request in order_requests:
                        order_limits = self.risk_manager.check_order_limits(
                            event.timestamp, len(active_orders)
                        )
                        
                        if order_limits['can_submit_order']:
                            order_id = self.execution.submit_order(request, event.timestamp)
                            if order_id:
                                self.risk_manager.record_order_submission(event.timestamp)
                                self.state.num_orders += 1
            
            elif risk_metrics.kill_switch_active:
                # Flatten position if kill switch is active
                position = self.execution.get_position(self.orderbook.symbol)
                if abs(position.quantity) > 0.1:
                    flatten_order_id = self.execution.flatten_position(
                        self.orderbook.symbol, event.timestamp
                    )
                    if flatten_order_id:
                        print(f"Flattening position due to kill switch: {position.quantity}")
            
            # Update state
            self._update_state()
            
            return True
            
        except Exception as e:
            print(f"Error processing event: {e}")
            return False
    
    def _extract_market_data(self, event: MarketEvent) -> Dict[str, Any]:
        """Extract market data from event"""
        market_data = {
            'timestamp': event.timestamp,
            'symbol': event.symbol
        }
        
        if isinstance(event, QuoteEvent):
            market_data.update({
                'bid': event.bid,
                'ask': event.ask,
                'bid_size': event.bid_size,
                'ask_size': event.ask_size,
                'mid_price': (event.bid + event.ask) / 2,
                'spread': event.ask - event.bid,
                'tick_size': self.config.get('simulation', {}).get('tick_size', 0.01)
            })
        elif isinstance(event, TradeEvent):
            market_data.update({
                'price': event.price,
                'size': event.size,
                'side': event.side
            })
            
            # Use last known quotes if available
            if self.orderbook.get_best_bid() and self.orderbook.get_best_ask():
                best_bid = self.orderbook.get_best_bid()
                best_ask = self.orderbook.get_best_ask()
                market_data.update({
                    'bid': best_bid[0],
                    'ask': best_ask[0],
                    'bid_size': best_bid[1],
                    'ask_size': best_ask[1],
                    'mid_price': self.orderbook.get_mid_price(),
                    'spread': self.orderbook.get_spread()
                })
        
        return market_data
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        position = self.execution.get_position(self.orderbook.symbol)
        portfolio_value = self.execution.get_portfolio_value()
        total_pnl = self.execution.get_total_pnl()
        
        return {
            'position': position.quantity,
            'portfolio_value': portfolio_value,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': total_pnl,
            'cash': self.execution.cash,
            'fees_paid': position.fees_paid,
            'rebates_earned': position.rebates_earned
        }
    
    def _get_time_to_close(self) -> float:
        """Get time remaining until close (0-1)"""
        # Simple linear decay based on steps
        max_steps = self._calculate_max_events()
        if max_steps > 0:
            return max(0.0, 1.0 - self.state.step / max_steps)
        return 0.5  # Default
    
    def _should_stop(self) -> bool:
        """Check if backtest should stop"""
        target_trades = self.config.get('backtest', {}).get('trade_target', 1500)
        
        # Stop if target trades reached
        if self.state.num_trades >= target_trades:
            return True
        
        # Stop if data feed is complete
        if self.datafeed.is_complete():
            return True
        
        # Stop if kill switch has been active too long
        if self.risk_manager.kill_switch_active:
            time_since_kill = self.state.timestamp - (self.risk_manager.kill_switch_timestamp or 0)
            if time_since_kill > 60:  # 1 minute
                return True
        
        return False
    
    def _update_state(self):
        """Update backtest state"""
        # Get current market price
        mid_price = self.orderbook.get_mid_price()
        if mid_price:
            self.state.current_price = mid_price
        
        spread = self.orderbook.get_spread()
        if spread:
            self.state.current_spread = spread
        
        # Update portfolio metrics
        portfolio_state = self._get_portfolio_state()
        self.state.cash = portfolio_state['cash']
        self.state.position = portfolio_state['position']
        self.state.portfolio_value = portfolio_state['portfolio_value']
        self.state.total_pnl = portfolio_state['total_pnl']
        
        # Update trade count
        self.state.num_trades = len(self.trade_log)
    
    def _log_trade(self, execution_report):
        """Log trade execution"""
        trade_data = {
            'timestamp': execution_report.timestamp,
            'order_id': execution_report.order_id,
            'side': execution_report.side.value,
            'quantity': execution_report.filled_qty,
            'price': execution_report.fill_price,
            'fees': execution_report.fees,
            'rebates': execution_report.rebates,
            'is_aggressive': execution_report.is_aggressive,
            'cumulative_qty': execution_report.cumulative_qty,
            'avg_price': execution_report.avg_price
        }
        self.trade_log.append(trade_data)
    
    def _log_signals(self, signals: Dict[str, Any]):
        """Log signal data"""
        self.signal_log.append(signals.copy())
    
    def _log_risk(self, risk_metrics):
        """Log risk metrics"""
        risk_data = {
            'timestamp': risk_metrics.timestamp,
            'position': risk_metrics.position,
            'portfolio_value': risk_metrics.portfolio_value,
            'total_pnl': risk_metrics.total_pnl,
            'current_drawdown': risk_metrics.current_drawdown,
            'max_drawdown': risk_metrics.max_drawdown,
            'var_1min': risk_metrics.var_1min,
            'risk_level': risk_metrics.risk_level.value,
            'kill_switch_active': risk_metrics.kill_switch_active
        }
        self.risk_log.append(risk_data)
    
    def _log_quote(self, quote):
        """Log quote data"""
        quote_data = {
            'timestamp': quote.timestamp,
            'bid_price': quote.bid_price,
            'ask_price': quote.ask_price,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
            'spread': quote.ask_price - quote.bid_price,
            'reservation_price': quote.reservation_price,
            'half_spread': quote.half_spread,
            'skew': quote.skew,
            'stat_arb_bias': quote.stat_arb_bias
        }
        self.quote_log.append(quote_data)
    
    def _finalize_backtest(self):
        """Finalize backtest and calculate results"""
        
        # Calculate final metrics
        final_metrics = self.metrics_calc.calculate_final_metrics(
            self.trade_log, self.risk_log, self.signal_log, self.quote_log
        )
        
        # Create results object
        self.results = BacktestResults(
            config=self.config,
            state=self.state,
            metrics=final_metrics,
            trade_log=self.trade_log,
            risk_log=self.risk_log,
            signal_log=self.signal_log,
            quote_log=self.quote_log,
            total_return=final_metrics.get('total_return', 0.0),
            sharpe_ratio=final_metrics.get('sharpe_ratio', 0.0),
            win_rate=final_metrics.get('win_rate', 0.0),
            max_drawdown=final_metrics.get('max_drawdown', 0.0),
            num_trades=len(self.trade_log)
        )
    
    def _print_progress(self):
        """Print progress update"""
        print(f"Step {self.state.step:,} | "
              f"Trades: {self.state.num_trades} | "
              f"P&L: ${self.state.total_pnl:,.0f} | "
              f"Position: {self.state.position:.1f} | "
              f"Price: ${self.state.current_price:.2f}")
    
    def _print_final_summary(self):
        """Print final summary"""
        if not self.results:
            return
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades:     {self.results.num_trades:,}")
        print(f"Total Return:     {self.results.total_return:.1%}")
        print(f"Sharpe Ratio:     {self.results.sharpe_ratio:.2f}")
        print(f"Win Rate:         {self.results.win_rate:.1%}")
        print(f"Max Drawdown:     {self.results.max_drawdown:.1%}")
        print(f"Final P&L:        ${self.state.total_pnl:,.0f}")
        print(f"Final Position:   {self.state.position:.1f}")
        print(f"Portfolio Value:  ${self.state.portfolio_value:,.0f}")
        
        # Risk summary
        risk_summary = self.risk_manager.get_risk_summary()
        if risk_summary:
            print(f"\nRisk Metrics:")
            print(f"  Kill Switch:    {'ACTIVE' if risk_summary['kill_switch_active'] else 'Inactive'}")
            print(f"  Position Util:  {risk_summary['position_utilization']:.1%}")
            print(f"  VaR (1min):     ${risk_summary['var_1min']:.0f}")
        
        print(f"{'='*60}")


def run_backtest(config_path: str, verbose: bool = True) -> BacktestResults:
    """
    Convenience function to run a backtest
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print progress and results
    
    Returns:
        BacktestResults object with all results and metrics
    """
    engine = BacktestEngine(config_path)
    return engine.run(verbose=verbose)


def run_parameter_sweep(base_config_path: str, parameter_grid: Dict[str, List[Any]], 
                       verbose: bool = False) -> List[BacktestResults]:
    """
    Run parameter sweep over multiple configurations
    
    Args:
        base_config_path: Base configuration file
        parameter_grid: Dictionary of parameters to sweep over
        verbose: Whether to print individual backtest results
    
    Returns:
        List of BacktestResults for each parameter combination
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate parameter combinations
    import itertools
    
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    
    results = []
    
    for combination in itertools.product(*param_values):
        # Create modified config
        config = base_config.copy()
        
        for param_name, param_value in zip(param_names, combination):
            # Handle nested parameters (e.g., "risk.gamma")
            keys = param_name.split('.')
            current_dict = config
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
            current_dict[keys[-1]] = param_value
        
        # Save temporary config
        temp_config_path = f"/tmp/temp_config_{len(results)}.yml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run backtest
            print(f"Running backtest {len(results)+1} with params: {dict(zip(param_names, combination))}")
            result = run_backtest(temp_config_path, verbose=verbose)
            results.append(result)
            
        except Exception as e:
            print(f"Backtest failed: {e}")
            continue
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python -m mm.backtest <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    results = run_backtest(config_path)
    
    print(f"\nBacktest completed. Results saved in results object.")
    print(f"Key metrics:")
    print(f"  Total Return: {results.total_return:.1%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Win Rate: {results.win_rate:.1%}")
    print(f"  Max DD: {results.max_drawdown:.1%}")
    print(f"  Trades: {results.num_trades}")
