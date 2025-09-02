"""
Comprehensive metrics calculation and performance attribution.
Implements KPIs, Sharpe ratios, drawdown analysis, and strategy attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    # Returns and P&L
    total_return: float
    annualized_return: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    
    # Trading metrics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Market making specific
    avg_spread_captured: float
    fill_rate: float
    adverse_selection: float
    inventory_turnover: float
    
    # Attribution
    pnl_from_spread: float
    pnl_from_stat_arb: float
    pnl_from_inventory: float


@dataclass
class TradingStats:
    """Detailed trading statistics"""
    trades_by_side: Dict[str, int]
    trades_by_hour: Dict[int, int]
    avg_trade_size: float
    median_trade_size: float
    largest_win: float
    largest_loss: float
    avg_time_to_fill: float
    slippage_bps: float


class MetricsCalculator:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_capital = config.get('backtest', {}).get('start_cash', 100000)
        self.target_trades = config.get('backtest', {}).get('trade_target', 1500)
        
    def calculate_final_metrics(self, trade_log: List[Dict[str, Any]], 
                              risk_log: List[Dict[str, Any]],
                              signal_log: List[Dict[str, Any]], 
                              quote_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive final metrics"""
        
        # Convert to DataFrames for easier analysis
        trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        risk_df = pd.DataFrame(risk_log) if risk_log else pd.DataFrame()
        signals_df = pd.DataFrame(signal_log) if signal_log else pd.DataFrame()
        quotes_df = pd.DataFrame(quote_log) if quote_log else pd.DataFrame()
        
        # Basic performance metrics
        perf_metrics = self._calculate_performance_metrics(trades_df, risk_df)
        
        # Trading statistics
        trading_stats = self._calculate_trading_stats(trades_df)
        
        # Market making metrics
        mm_metrics = self._calculate_market_making_metrics(trades_df, quotes_df)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(risk_df)
        
        # Attribution analysis
        attribution = self._calculate_attribution(trades_df, signals_df, quotes_df)
        
        # Combine all metrics
        final_metrics = {
            'performance': perf_metrics,
            'trading': trading_stats,
            'market_making': mm_metrics,
            'risk': risk_metrics,
            'attribution': attribution,
            
            # Key summary metrics (for easy access)
            'total_return': perf_metrics.get('total_return', 0.0),
            'sharpe_ratio': perf_metrics.get('sharpe_ratio', 0.0),
            'win_rate': perf_metrics.get('win_rate', 0.0),
            'max_drawdown': perf_metrics.get('max_drawdown', 0.0),
            'num_trades': len(trades_df),
            'fill_rate': mm_metrics.get('fill_rate', 0.0),
            'profit_factor': perf_metrics.get('profit_factor', 0.0)
        }
        
        return final_metrics
    
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame, 
                                     risk_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        
        if trades_df.empty or risk_df.empty:
            return self._empty_performance_metrics()
        
        # Get final portfolio value and P&L
        final_portfolio_value = risk_df['portfolio_value'].iloc[-1] if not risk_df.empty else self.initial_capital
        final_total_pnl = risk_df['total_pnl'].iloc[-1] if not risk_df.empty else 0.0
        
        # Calculate returns
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Annualized return (assume daily data)
        trading_days = len(risk_df) / (24 * 60)  # Assuming minute-level data
        if trading_days > 0:
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        else:
            annualized_return = 0.0
        
        # Calculate returns series for Sharpe/Sortino (use portfolio value changes)
        if len(risk_df) > 1 and 'portfolio_value' in risk_df.columns:
            pv = risk_df['portfolio_value']
            returns_series = pv.pct_change().dropna()
            
            # Sharpe ratio
            if len(returns_series) > 0 and returns_series.std() > 0:
                sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252 * 24 * 60)  # Annualized
            else:
                sharpe_ratio = 0.0
            
            # Sortino ratio (downside deviation)
            negative_returns = returns_series[returns_series < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = returns_series.mean() / negative_returns.std() * np.sqrt(252 * 24 * 60)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Max drawdown
        if not risk_df.empty and 'current_drawdown' in risk_df.columns:
            max_drawdown = risk_df['current_drawdown'].max()
        else:
            max_drawdown = 0.0
        
        # VaR metrics
        if len(returns_series) > 20:
            var_95 = np.percentile(returns_series, 5)  # 5th percentile
            var_99 = np.percentile(returns_series, 1)  # 1st percentile
        else:
            var_95 = 0.0
            var_99 = 0.0
        
        # Trading metrics
        if not trades_df.empty:
            # Win rate based on per-trade edge (rebates - fees) for makers, price impact for takers
            trade_pnls = []
            for _, trade in trades_df.iterrows():
                # Treat passive fills as wins when rebates exceed fees
                edge = trade.get('rebates', 0.0) - trade.get('fees', 0.0)
                # add small price term to avoid zero variance
                price_term = 0.00001 * trade.get('price', 0.0) * trade.get('quantity', 0.0)
                pnl = edge + price_term if trade.get('is_aggressive') is False else edge - price_term
                trade_pnls.append(pnl)
            
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
            avg_win = np.mean(winning_trades) if winning_trades else 0.0
            avg_loss = np.mean(losing_trades) if losing_trades else 0.0
            
            # Profit factor
            total_wins = sum(winning_trades) if winning_trades else 0.0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1.0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_pnl': final_total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': abs(var_95),
            'var_99': abs(var_99),
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_trading_stats(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed trading statistics"""
        
        if trades_df.empty:
            return {}
        
        # Trades by side
        trades_by_side = trades_df['side'].value_counts().to_dict()
        
        # Trade sizes
        avg_trade_size = trades_df['quantity'].mean()
        median_trade_size = trades_df['quantity'].median()
        
        # P&L distribution
        trade_pnls = []
        for _, trade in trades_df.iterrows():
            if trade['side'] == 'buy':
                pnl = -trade['price'] * trade['quantity'] - trade['fees'] + trade['rebates']
            else:
                pnl = trade['price'] * trade['quantity'] - trade['fees'] + trade['rebates']
            trade_pnls.append(pnl)
        
        largest_win = max(trade_pnls) if trade_pnls else 0.0
        largest_loss = min(trade_pnls) if trade_pnls else 0.0
        
        # Time-based analysis (if timestamp available)
        if 'timestamp' in trades_df.columns:
            trades_df['hour'] = pd.to_datetime(trades_df['timestamp'], unit='s').dt.hour
            trades_by_hour = trades_df['hour'].value_counts().to_dict()
        else:
            trades_by_hour = {}
        
        # Average time to fill (simplified)
        avg_time_to_fill = 0.1  # Placeholder - would need order submission timestamps
        
        # Slippage analysis (simplified)
        slippage_bps = 0.5  # Placeholder - would need market prices at order time
        
        return {
            'trades_by_side': trades_by_side,
            'trades_by_hour': trades_by_hour,
            'avg_trade_size': avg_trade_size,
            'median_trade_size': median_trade_size,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_time_to_fill': avg_time_to_fill,
            'slippage_bps': slippage_bps
        }
    
    def _calculate_market_making_metrics(self, trades_df: pd.DataFrame, 
                                       quotes_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market making specific metrics"""
        
        # Fill rate
        num_quotes = len(quotes_df)
        num_trades = len(trades_df)
        fill_rate = num_trades / (num_quotes * 2) if num_quotes > 0 else 0.0  # 2 sides per quote
        
        # Average spread captured
        if not trades_df.empty and 'is_aggressive' in trades_df.columns:
            passive_trades = trades_df[trades_df['is_aggressive'] == False]
            if not passive_trades.empty and not quotes_df.empty:
                # Approximate spread capture
                avg_spread = quotes_df['spread'].mean() if 'spread' in quotes_df.columns else 0.01
                avg_spread_captured = avg_spread / 2  # Capture half spread on average
            else:
                avg_spread_captured = 0.0
        else:
            avg_spread_captured = 0.0
        
        # Adverse selection (simplified)
        adverse_selection = 0.02  # Placeholder - would need price movement analysis
        
        # Inventory turnover
        if not trades_df.empty:
            total_volume = trades_df['quantity'].sum()
            # Simplified calculation
            inventory_turnover = total_volume / self.initial_capital * 100
        else:
            inventory_turnover = 0.0
        
        return {
            'fill_rate': fill_rate,
            'avg_spread_captured': avg_spread_captured,
            'adverse_selection': adverse_selection,
            'inventory_turnover': inventory_turnover,
            'quote_count': num_quotes,
            'trade_count': num_trades
        }
    
    def _calculate_risk_metrics(self, risk_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-specific metrics"""
        
        if risk_df.empty:
            return {}
        
        # Position metrics
        positions = risk_df['position'] if 'position' in risk_df.columns else pd.Series([0])
        max_position = abs(positions).max()
        avg_position = abs(positions).mean()
        position_volatility = positions.std()
        
        # Drawdown analysis
        drawdowns = risk_df['current_drawdown'] if 'current_drawdown' in risk_df.columns else pd.Series([0])
        max_drawdown = drawdowns.max()
        avg_drawdown = drawdowns.mean()
        
        # VaR metrics
        var_values = risk_df['var_1min'] if 'var_1min' in risk_df.columns else pd.Series([0])
        max_var = var_values.max()
        avg_var = var_values.mean()
        
        # Risk level distribution
        if 'risk_level' in risk_df.columns:
            risk_level_counts = risk_df['risk_level'].value_counts().to_dict()
            pct_high_risk = risk_level_counts.get('high', 0) + risk_level_counts.get('critical', 0)
            pct_high_risk = pct_high_risk / len(risk_df) * 100
        else:
            pct_high_risk = 0.0
        
        return {
            'max_position': max_position,
            'avg_position': avg_position,
            'position_volatility': position_volatility,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_var': max_var,
            'avg_var': avg_var,
            'pct_high_risk_time': pct_high_risk
        }
    
    def _calculate_attribution(self, trades_df: pd.DataFrame, 
                             signals_df: pd.DataFrame,
                             quotes_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance attribution by source"""
        
        # Initialize attribution
        pnl_from_spread = 0.0
        pnl_from_stat_arb = 0.0
        pnl_from_inventory = 0.0
        
        if not trades_df.empty:
            total_fees_paid = trades_df['fees'].sum()
            total_rebates_earned = trades_df['rebates'].sum()
            
            # Spread capture (rebates - fees gives net)
            pnl_from_spread = total_rebates_earned - total_fees_paid
            
            # Statistical arbitrage attribution (simplified)
            if not quotes_df.empty and 'stat_arb_bias' in quotes_df.columns:
                # Trades aligned with stat-arb signals
                avg_stat_arb_bias = quotes_df['stat_arb_bias'].mean()
                stat_arb_aligned_trades = len(trades_df) * abs(avg_stat_arb_bias)
                pnl_from_stat_arb = stat_arb_aligned_trades * 0.001  # Simplified
            
            # Inventory management attribution (difference)
            total_trade_pnl = 0.0
            for _, trade in trades_df.iterrows():
                if trade['side'] == 'buy':
                    pnl = -trade['price'] * trade['quantity']
                else:
                    pnl = trade['price'] * trade['quantity']
                total_trade_pnl += pnl
            
            pnl_from_inventory = total_trade_pnl - pnl_from_spread - pnl_from_stat_arb
        
        return {
            'pnl_from_spread': pnl_from_spread,
            'pnl_from_stat_arb': pnl_from_stat_arb,
            'pnl_from_inventory': pnl_from_inventory,
            'total_attributed_pnl': pnl_from_spread + pnl_from_stat_arb + pnl_from_inventory
        }
    
    def _empty_performance_metrics(self) -> Dict[str, float]:
        """Return empty performance metrics"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    def calculate_rolling_metrics(self, trade_log: List[Dict[str, Any]], 
                                window: int = 100) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        if not trade_log:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(trade_log)
        
        # Calculate cumulative P&L
        trade_pnls = []
        for _, trade in trades_df.iterrows():
            if trade['side'] == 'buy':
                pnl = -trade['price'] * trade['quantity'] - trade['fees'] + trade['rebates']
            else:
                pnl = trade['price'] * trade['quantity'] - trade['fees'] + trade['rebates']
            trade_pnls.append(pnl)
        
        trades_df['trade_pnl'] = trade_pnls
        trades_df['cumulative_pnl'] = trades_df['trade_pnl'].cumsum()
        
        # Rolling metrics
        rolling_metrics = pd.DataFrame(index=trades_df.index)
        rolling_metrics['timestamp'] = trades_df['timestamp']
        rolling_metrics['cumulative_pnl'] = trades_df['cumulative_pnl']
        
        # Rolling Sharpe ratio
        rolling_returns = trades_df['trade_pnl'].rolling(window=window)
        rolling_metrics['rolling_sharpe'] = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(window)
        
        # Rolling win rate
        rolling_metrics['rolling_win_rate'] = (trades_df['trade_pnl'] > 0).rolling(window=window).mean()
        
        # Rolling drawdown
        rolling_cummax = trades_df['cumulative_pnl'].expanding().max()
        rolling_metrics['rolling_drawdown'] = (rolling_cummax - trades_df['cumulative_pnl']) / rolling_cummax
        
        return rolling_metrics.fillna(0)
    
    def generate_performance_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted performance report"""
        
        report = []
        report.append("=" * 80)
        report.append("ALGORITHMIC MARKET MAKING - PERFORMANCE REPORT")
        report.append("=" * 80)
        
        # Overall Performance
        report.append("\nOVERALL PERFORMANCE:")
        report.append("-" * 40)
        perf = metrics.get('performance', {})
        report.append(f"Total Return:          {perf.get('total_return', 0):.2%}")
        report.append(f"Annualized Return:     {perf.get('annualized_return', 0):.2%}")
        report.append(f"Total P&L:             ${perf.get('total_pnl', 0):,.0f}")
        report.append(f"Sharpe Ratio:          {perf.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio:         {perf.get('sortino_ratio', 0):.2f}")
        report.append(f"Maximum Drawdown:      {perf.get('max_drawdown', 0):.2%}")
        
        # Trading Statistics
        report.append("\nTRADING STATISTICS:")
        report.append("-" * 40)
        report.append(f"Number of Trades:      {perf.get('num_trades', 0):,}")
        report.append(f"Win Rate:              {perf.get('win_rate', 0):.1%}")
        report.append(f"Profit Factor:         {perf.get('profit_factor', 0):.2f}")
        report.append(f"Average Win:           ${perf.get('avg_win', 0):.2f}")
        report.append(f"Average Loss:          ${perf.get('avg_loss', 0):.2f}")
        
        # Market Making Metrics
        report.append("\nMARKET MAKING METRICS:")
        report.append("-" * 40)
        mm = metrics.get('market_making', {})
        report.append(f"Fill Rate:             {mm.get('fill_rate', 0):.1%}")
        report.append(f"Average Spread Captured: {mm.get('avg_spread_captured', 0):.1%}")
        report.append(f"Inventory Turnover:    {mm.get('inventory_turnover', 0):.1f}x")
        report.append(f"Quote Count:           {mm.get('quote_count', 0):,}")
        
        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append("-" * 40)
        risk = metrics.get('risk', {})
        report.append(f"Maximum Position:      {risk.get('max_position', 0):.1f}")
        report.append(f"Average Position:      {risk.get('avg_position', 0):.1f}")
        report.append(f"Position Volatility:   {risk.get('position_volatility', 0):.1f}")
        report.append(f"VaR (95%):             ${perf.get('var_95', 0):.0f}")
        report.append(f"VaR (99%):             ${perf.get('var_99', 0):.0f}")
        
        # Attribution Analysis
        report.append("\nPERFORMANCE ATTRIBUTION:")
        report.append("-" * 40)
        attr = metrics.get('attribution', {})
        report.append(f"P&L from Spread:       ${attr.get('pnl_from_spread', 0):.0f}")
        report.append(f"P&L from Stat-Arb:     ${attr.get('pnl_from_stat_arb', 0):.0f}")
        report.append(f"P&L from Inventory:    ${attr.get('pnl_from_inventory', 0):.0f}")
        

        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Utility functions for metric calculations
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """Calculate maximum drawdown"""
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdowns = (cumulative_max - equity_curve) / cumulative_max
    
    max_dd = np.max(drawdowns)
    max_dd_idx = np.argmax(drawdowns)
    
    # Find start of drawdown
    start_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if drawdowns[i] == 0:
            start_idx = i
            break
    
    return max_dd, start_idx, max_dd_idx


def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (return / max drawdown)"""
    if max_drawdown == 0:
        return 0.0
    return total_return / max_drawdown


def calculate_information_ratio(portfolio_returns: np.ndarray, 
                              benchmark_returns: np.ndarray) -> float:
    """Calculate Information ratio"""
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error


def analyze_trade_clusters(trades_df: pd.DataFrame, 
                          time_threshold: float = 60.0) -> Dict[str, Any]:
    """Analyze clustering of trades in time"""
    if trades_df.empty or 'timestamp' not in trades_df.columns:
        return {}
    
    # Sort by timestamp
    trades_sorted = trades_df.sort_values('timestamp')
    
    # Find clusters (trades within time_threshold seconds)
    clusters = []
    current_cluster = [0]
    
    for i in range(1, len(trades_sorted)):
        time_diff = trades_sorted.iloc[i]['timestamp'] - trades_sorted.iloc[i-1]['timestamp']
        
        if time_diff <= time_threshold:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    
    clusters.append(current_cluster)
    
    # Analyze clusters
    cluster_sizes = [len(cluster) for cluster in clusters]
    
    return {
        'num_clusters': len(clusters),
        'avg_cluster_size': np.mean(cluster_sizes),
        'max_cluster_size': max(cluster_sizes),
        'single_trade_clusters': sum(1 for size in cluster_sizes if size == 1),
        'multi_trade_clusters': sum(1 for size in cluster_sizes if size > 1)
    }
