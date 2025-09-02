"""Risk management system with position limits and drawdown controls."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: float
    position: float
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    max_drawdown: float
    current_drawdown: float
    var_1min: float
    position_limit_utilization: float
    risk_level: RiskLevel
    
    # Risk flags
    kill_switch_active: bool = False
    position_limit_breach: bool = False
    drawdown_limit_breach: bool = False
    var_limit_breach: bool = False


@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    current_value: float
    limit_value: float
    threshold_pct: float = 0.8  # Warning at 80% of limit
    
    @property
    def utilization(self) -> float:
        if self.limit_value == 0:
            return 0.0
        return abs(self.current_value) / abs(self.limit_value)
    
    @property
    def is_warning(self) -> bool:
        return self.utilization >= self.threshold_pct
    
    @property
    def is_breach(self) -> bool:
        return self.utilization >= 1.0


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits
        self.max_position = config.get('q_max', 500)
        self.max_drawdown_pct = config.get('dd_stop_pct', 0.03)  # 3%
        self.var_limit = config.get('var_limit', 1000)
        self.max_leverage = config.get('max_leverage', 3.0)
        
        # Rate limits
        self.max_orders_per_minute = config.get('max_orders_per_minute', 60)
        self.max_open_orders = config.get('max_open_orders', 10)
        
        # Portfolio tracking
        self.initial_capital = config.get('start_cash', 100000)
        self.peak_portfolio_value = self.initial_capital
        self.max_drawdown_reached = 0.0
        
        # Risk state
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_timestamp = None
        
        # Risk history
        self.risk_history: List[RiskMetrics] = []
        self.pnl_history = deque(maxlen=1000)  # For VaR calculation
        self.position_history = deque(maxlen=100)
        
        # Time-based tracking
        self.order_timestamps = deque(maxlen=100)
        
        # VaR calculation
        self.var_lookback = 60  # 1 minute for 1-min VaR
        self.var_confidence = 0.95  # 95% VaR
        
    def update_risk_metrics(self, portfolio_state: Dict[str, Any], 
                          current_time: float) -> RiskMetrics:
        """Update and return current risk metrics"""
        
        # Extract portfolio state
        position = portfolio_state.get('position', 0.0)
        portfolio_value = portfolio_state.get('portfolio_value', self.initial_capital)
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        realized_pnl = portfolio_state.get('realized_pnl', 0.0)
        total_pnl = realized_pnl + unrealized_pnl
        
        # Update position history
        self.position_history.append(position)
        
        # Update P&L history for VaR
        if self.pnl_history:
            pnl_change = total_pnl - self.pnl_history[-1]
        else:
            pnl_change = 0.0
        self.pnl_history.append(total_pnl)
        
        # Calculate drawdown
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        self.max_drawdown_reached = max(self.max_drawdown_reached, current_drawdown)
        
        # Calculate VaR
        var_1min = self._calculate_var()
        
        # Position limit utilization
        position_limit_util = abs(position) / self.max_position if self.max_position > 0 else 0
        
        # Determine risk level
        risk_level = self._assess_risk_level(current_drawdown, position_limit_util, var_1min)
        
        # Check for limit breaches
        position_breach = abs(position) > self.max_position
        drawdown_breach = current_drawdown > self.max_drawdown_pct
        var_breach = var_1min > self.var_limit
        
        # Update kill switch
        if not self.kill_switch_active:
            if drawdown_breach:
                self._activate_kill_switch(f"Drawdown breach: {current_drawdown:.1%} > {self.max_drawdown_pct:.1%}", current_time)
            elif var_breach:
                self._activate_kill_switch(f"VaR breach: {var_1min:.0f} > {self.var_limit:.0f}", current_time)
            elif position_breach:
                self._activate_kill_switch(f"Position breach: {abs(position):.0f} > {self.max_position:.0f}", current_time)
        
        # Create risk metrics
        metrics = RiskMetrics(
            timestamp=current_time,
            position=position,
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            max_drawdown=self.max_drawdown_reached,
            current_drawdown=current_drawdown,
            var_1min=var_1min,
            position_limit_utilization=position_limit_util,
            risk_level=risk_level,
            kill_switch_active=self.kill_switch_active,
            position_limit_breach=position_breach,
            drawdown_limit_breach=drawdown_breach,
            var_limit_breach=var_breach
        )
        
        # Store in history
        self.risk_history.append(metrics)
        
        # Keep limited history
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
        
        return metrics
    
    def _calculate_var(self) -> float:
        """Calculate 1-minute Value at Risk"""
        if len(self.pnl_history) < 10:
            return 0.0
        
        # Get recent P&L changes
        recent_pnl = list(self.pnl_history)[-self.var_lookback:]
        pnl_changes = np.diff(recent_pnl) if len(recent_pnl) > 1 else [0.0]
        
        if len(pnl_changes) < 5:
            return 0.0
        
        # Calculate VaR as percentile of P&L distribution
        var_percentile = (1 - self.var_confidence) * 100
        var_value = np.percentile(pnl_changes, var_percentile)
        
        # Return absolute value (loss)
        return abs(var_value)
    
    def _assess_risk_level(self, drawdown: float, position_util: float, var: float) -> RiskLevel:
        """Assess overall risk level"""
        
        # Count risk factors
        risk_factors = 0
        
        if drawdown > self.max_drawdown_pct * 0.5:  # 50% of DD limit
            risk_factors += 1
        if drawdown > self.max_drawdown_pct * 0.8:  # 80% of DD limit
            risk_factors += 1
        
        if position_util > 0.5:  # 50% of position limit
            risk_factors += 1
        if position_util > 0.8:  # 80% of position limit
            risk_factors += 1
        
        if var > self.var_limit * 0.5:  # 50% of VaR limit
            risk_factors += 1
        if var > self.var_limit * 0.8:  # 80% of VaR limit
            risk_factors += 1
        
        # Determine level
        if risk_factors >= 4:
            return RiskLevel.CRITICAL
        elif risk_factors >= 3:
            return RiskLevel.HIGH
        elif risk_factors >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _activate_kill_switch(self, reason: str, timestamp: float):
        """Activate kill switch"""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_timestamp = timestamp
        print(f"KILL SWITCH ACTIVATED: {reason} at {timestamp}")
    
    def deactivate_kill_switch(self):
        """Manually deactivate kill switch (use with caution)"""
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_timestamp = None
        print("Kill switch deactivated")
    
    def check_order_limits(self, current_time: float, num_open_orders: int) -> Dict[str, Any]:
        """Check if new orders are allowed"""
        
        # Clean old order timestamps
        cutoff_time = current_time - 60.0  # 1 minute
        while self.order_timestamps and self.order_timestamps[0] < cutoff_time:
            self.order_timestamps.popleft()
        
        # Check rate limits
        orders_in_last_minute = len(self.order_timestamps)
        rate_limit_ok = orders_in_last_minute < self.max_orders_per_minute
        
        # Check open order limit
        open_orders_ok = num_open_orders < self.max_open_orders
        
        # Check kill switch
        kill_switch_ok = not self.kill_switch_active
        
        result = {
            'can_submit_order': rate_limit_ok and open_orders_ok and kill_switch_ok,
            'rate_limit_ok': rate_limit_ok,
            'open_orders_ok': open_orders_ok,
            'kill_switch_ok': kill_switch_ok,
            'orders_in_last_minute': orders_in_last_minute,
            'num_open_orders': num_open_orders
        }
        
        return result
    
    def record_order_submission(self, timestamp: float):
        """Record order submission for rate limiting"""
        self.order_timestamps.append(timestamp)
    
    def should_reduce_position(self, current_position: float) -> Tuple[bool, float]:
        """Check if position should be reduced and by how much"""
        
        if self.kill_switch_active:
            # Flatten completely
            return True, -current_position
        
        # Check if position is too large
        if abs(current_position) > self.max_position * 0.9:  # 90% of limit
            # Reduce to 50% of limit
            target_position = np.sign(current_position) * self.max_position * 0.5
            reduction = target_position - current_position
            return True, reduction
        
        return False, 0.0
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk state"""
        if not self.risk_history:
            return {}
        
        latest = self.risk_history[-1]
        
        # Calculate additional metrics
        avg_position = np.mean(list(self.position_history)) if self.position_history else 0
        position_volatility = np.std(list(self.position_history)) if len(self.position_history) > 1 else 0
        
        summary = {
            'timestamp': latest.timestamp,
            'risk_level': latest.risk_level.value,
            'kill_switch_active': latest.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            
            # Position metrics
            'current_position': latest.position,
            'position_limit': self.max_position,
            'position_utilization': latest.position_limit_utilization,
            'avg_position': avg_position,
            'position_volatility': position_volatility,
            
            # P&L metrics
            'total_pnl': latest.total_pnl,
            'realized_pnl': latest.realized_pnl,
            'unrealized_pnl': latest.unrealized_pnl,
            'portfolio_value': latest.portfolio_value,
            
            # Risk metrics
            'current_drawdown': latest.current_drawdown,
            'max_drawdown_reached': latest.max_drawdown,
            'max_drawdown_limit': self.max_drawdown_pct,
            'var_1min': latest.var_1min,
            'var_limit': self.var_limit,
            
            # Limits status
            'limits': {
                'position': {
                    'current': abs(latest.position),
                    'limit': self.max_position,
                    'utilization': latest.position_limit_utilization,
                    'breach': latest.position_limit_breach
                },
                'drawdown': {
                    'current': latest.current_drawdown,
                    'limit': self.max_drawdown_pct,
                    'utilization': latest.current_drawdown / self.max_drawdown_pct if self.max_drawdown_pct > 0 else 0,
                    'breach': latest.drawdown_limit_breach
                },
                'var': {
                    'current': latest.var_1min,
                    'limit': self.var_limit,
                    'utilization': latest.var_1min / self.var_limit if self.var_limit > 0 else 0,
                    'breach': latest.var_limit_breach
                }
            }
        }
        
        return summary
    
    def get_position_sizing_adjustment(self, base_size: float, current_position: float, 
                                     risk_level: RiskLevel) -> float:
        """Adjust position size based on current risk"""
        
        if self.kill_switch_active:
            return 0.0  # No new positions
        
        # Base adjustment by risk level
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.CRITICAL: 0.1
        }
        
        risk_adj = risk_multipliers.get(risk_level, 0.5)
        
        # Additional adjustment based on position utilization
        position_util = abs(current_position) / self.max_position if self.max_position > 0 else 0
        if position_util > 0.8:
            position_adj = 0.2
        elif position_util > 0.6:
            position_adj = 0.5
        elif position_util > 0.4:
            position_adj = 0.8
        else:
            position_adj = 1.0
        
        # Combined adjustment
        total_adj = risk_adj * position_adj
        
        return base_size * total_adj
    
    def check_trading_halt_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if trading should be halted due to market conditions"""
        
        halt_conditions = {
            'should_halt': False,
            'reasons': []
        }
        
        # Check spread conditions
        spread = market_data.get('spread')
        if spread and spread > 0.05:  # 5% spread
            halt_conditions['should_halt'] = True
            halt_conditions['reasons'].append(f"Wide spread: {spread:.1%}")
        
        # Check volatility conditions
        volatility = market_data.get('volatility')
        if volatility and volatility > 0.1:  # 10% volatility
            halt_conditions['should_halt'] = True
            halt_conditions['reasons'].append(f"High volatility: {volatility:.1%}")
        
        # Check if market data is stale
        data_timestamp = market_data.get('timestamp', 0)
        current_time = time.time()
        if current_time - data_timestamp > 30:  # 30 seconds stale
            halt_conditions['should_halt'] = True
            halt_conditions['reasons'].append("Stale market data")
        
        return halt_conditions
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get current risk alerts"""
        alerts = []
        
        if not self.risk_history:
            return alerts
        
        latest = self.risk_history[-1]
        
        # Kill switch alert
        if latest.kill_switch_active:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'kill_switch',
                'message': f"Kill switch active: {self.kill_switch_reason}",
                'timestamp': self.kill_switch_timestamp
            })
        
        # Position alerts
        if latest.position_limit_utilization > 0.8:
            alerts.append({
                'level': 'HIGH',
                'type': 'position_limit',
                'message': f"Position limit {latest.position_limit_utilization:.1%} utilized",
                'timestamp': latest.timestamp
            })
        
        # Drawdown alerts
        if latest.current_drawdown > self.max_drawdown_pct * 0.8:
            alerts.append({
                'level': 'HIGH',
                'type': 'drawdown',
                'message': f"Drawdown {latest.current_drawdown:.1%} near limit {self.max_drawdown_pct:.1%}",
                'timestamp': latest.timestamp
            })
        
        # VaR alerts
        if latest.var_1min > self.var_limit * 0.8:
            alerts.append({
                'level': 'MEDIUM',
                'type': 'var',
                'message': f"VaR {latest.var_1min:.0f} near limit {self.var_limit:.0f}",
                'timestamp': latest.timestamp
            })
        
        return alerts
    
    def reset_kill_switch_after_flatten(self, portfolio_value: float):
        """Reset kill switch after position has been flattened"""
        if self.kill_switch_active and abs(portfolio_value - self.initial_capital) < self.initial_capital * 0.01:
            # Position is essentially flat, safe to reset
            self.deactivate_kill_switch()
            print("Kill switch auto-reset after position flattening")


# Utility functions for risk calculations
def calculate_portfolio_var(returns: np.ndarray, confidence: float = 0.95, 
                           lookback: int = 100) -> float:
    """Calculate portfolio Value at Risk"""
    if len(returns) < 10:
        return 0.0
    
    recent_returns = returns[-lookback:] if len(returns) > lookback else returns
    var_percentile = (1 - confidence) * 100
    
    return abs(np.percentile(recent_returns, var_percentile))


def calculate_maximum_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its start/end indices
    
    Returns:
        (max_drawdown, start_idx, end_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdowns = (cumulative_max - equity_curve) / cumulative_max
    
    max_dd = np.max(drawdowns)
    max_dd_idx = np.argmax(drawdowns)
    
    # Find start of drawdown (last peak before max drawdown)
    start_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if drawdowns[i] == 0:  # At a peak
            start_idx = i
            break
    
    return max_dd, start_idx, max_dd_idx


def position_risk_score(position: float, max_position: float, volatility: float) -> float:
    """Calculate position risk score (0-1)"""
    if max_position <= 0:
        return 0.0
    
    # Base risk from position size
    size_risk = abs(position) / max_position
    
    # Volatility adjustment
    vol_risk = min(volatility * 10, 1.0)  # Cap at 100%
    
    # Combined risk score
    total_risk = min(size_risk * (1 + vol_risk), 1.0)
    
    return total_risk
