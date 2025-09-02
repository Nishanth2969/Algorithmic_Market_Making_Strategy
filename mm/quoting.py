"""Inventory-aware quoting engine with Avellaneda-Stoikov framework."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .execution import OrderRequest, Side, OrderType


@dataclass
class Quote:

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: float
    reservation_price: float
    half_spread: float
    skew: float
    stat_arb_bias: float = 0.0


class QuotingEngine:
    """Inventory-aware market making engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk parameters
        self.gamma = config.get('gamma', 0.005)  # Risk aversion
        self.phi = config.get('inv_penalty_phi', 0.02)  # Inventory penalty
        
        # Arrival model parameters
        self.A = config.get('A', 1.8)  # Base arrival intensity
        self.k = config.get('k', 1.2)  # Spread sensitivity
        
        # Sizing parameters
        self.base_qty = config.get('base_qty', 5)
        self.vol_scale = config.get('vol_scale', True)
        self.max_leverage = config.get('max_leverage', 3.0)
        self.min_spread_bps = config.get('min_spread_bps', 2)
        
        # Stat-arb parameters
        self.max_bias = config.get('max_bias', 1.0)
        
        # State tracking
        self.last_quote: Optional[Quote] = None
        self.quote_history: List[Quote] = []
        
    def generate_quote(self, 
                      market_data: Dict[str, Any],
                      position: float,
                      signals: Dict[str, Any],
                      time_to_close: float = 1.0) -> Optional[Quote]:
        """
        Generate optimal bid/ask quotes using Avellaneda-Stoikov framework
        
        Args:
            market_data: Current market state
            position: Current inventory (signed quantity)
            signals: Signal dictionary from SignalManager
            time_to_close: Remaining time fraction (0-1)
        """
        # Extract market data
        fair_value = signals.get('microprice') or signals.get('mid_price')
        if fair_value is None:
            return None
        
        volatility = signals.get('volatility', 0.01)
        imbalance = signals.get('imbalance', 0.0)
        
        # Calculate reservation price (inventory-adjusted fair value)
        tau = time_to_close
        reservation_price = self._calculate_reservation_price(
            fair_value, position, volatility, tau
        )
        
        # Calculate optimal half-spread
        half_spread = self._calculate_optimal_spread(
            volatility, tau, position
        )
        
        # Apply minimum spread constraint
        min_spread = fair_value * self.min_spread_bps / 10000.0
        # Also cap maximum spread to avoid quoting too wide
        max_spread_bps = self.config.get('max_spread_bps', 50)  # 50 bps default
        max_spread = fair_value * max_spread_bps / 10000.0
        half_spread = np.clip(half_spread, min_spread / 2, max_spread / 2)
        
        # Calculate inventory skew
        skew = self._calculate_inventory_skew(position, volatility, tau)
        
        # Apply statistical arbitrage bias
        stat_arb_bias = self._calculate_stat_arb_bias(signals)
        
        # Calculate quote sizes
        bid_size, ask_size = self._calculate_quote_sizes(
            position, volatility, signals
        )
        
        # Generate final quotes
        bid_price = reservation_price - half_spread - skew - stat_arb_bias * half_spread
        ask_price = reservation_price + half_spread - skew - stat_arb_bias * half_spread
        
        # Round to tick size
        tick_size = market_data.get('tick_size', 0.01)
        bid_price = self._round_to_tick(bid_price, tick_size)
        ask_price = self._round_to_tick(ask_price, tick_size)
        
        # Ensure positive spread
        if ask_price <= bid_price:
            spread = max(tick_size, min_spread)
            mid = (bid_price + ask_price) / 2
            bid_price = mid - spread / 2
            ask_price = mid + spread / 2
            bid_price = self._round_to_tick(bid_price, tick_size)
            ask_price = self._round_to_tick(ask_price, tick_size)
        
        quote = Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=signals.get('timestamp', 0),
            reservation_price=reservation_price,
            half_spread=half_spread,
            skew=skew,
            stat_arb_bias=stat_arb_bias
        )
        
        self.last_quote = quote
        self.quote_history.append(quote)
        
        # Keep limited history
        if len(self.quote_history) > 1000:
            self.quote_history = self.quote_history[-1000:]
        
        return quote
    
    def _calculate_reservation_price(self, fair_value: float, position: float, 
                                   volatility: float, tau: float) -> float:
        """
        Calculate reservation price: r_t = s_t - q_t * γ * σ^2 * (T-t)
        
        This is the inventory-adjusted fair value that accounts for risk aversion
        """
        inventory_adjustment = position * self.gamma * (volatility ** 2) * tau
        return fair_value - inventory_adjustment
    
    def _calculate_optimal_spread(self, volatility: float, tau: float, position: float) -> float:
        """
        Calculate optimal half-spread using Avellaneda-Stoikov formula:
        δ* ≈ (1/γ) * ln(1 + γ/k) + (γ * σ^2 * τ) / 2
        """
        # Base spread from arrival rate optimization
        if self.gamma > 0 and self.k > 0:
            base_spread = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        else:
            base_spread = 0.01  # Fallback
        
        # Time and volatility adjustment
        vol_adjustment = 0.5 * self.gamma * (volatility ** 2) * tau
        
        # Position-dependent spread widening
        position_factor = 1.0 + 0.1 * abs(position) / 100.0  # Widen spread with larger positions
        
        optimal_spread = (base_spread + vol_adjustment) * position_factor
        
        return max(optimal_spread, 0.001)  # Minimum spread
    
    def _calculate_inventory_skew(self, position: float, volatility: float, tau: float) -> float:
        """
        Calculate inventory skew: bid/ask adjustment based on current position
        Positive skew moves both quotes up (encouraging sells)
        Negative skew moves both quotes down (encouraging buys)
        """
        # Basic linear skew
        linear_skew = self.phi * position
        
        # Increase skew with time pressure and volatility
        urgency_factor = 1.0 + (1.0 - tau) * 2.0  # More urgent near close
        vol_factor = 1.0 + volatility * 10.0  # Higher skew in volatile markets
        
        total_skew = linear_skew * urgency_factor * vol_factor
        
        # Cap the skew to prevent excessive quote movement
        max_skew = 0.05  # 5% max skew
        return np.clip(total_skew, -max_skew, max_skew)
    
    def _calculate_stat_arb_bias(self, signals: Dict[str, Any]) -> float:
        """
        Calculate statistical arbitrage bias from cointegration signals
        
        Returns bias in [-max_bias, max_bias] that adjusts quote symmetry
        Positive bias favors selling (moves quotes up)
        Negative bias favors buying (moves quotes down)
        """
        total_bias = 0.0
        signal_count = 0
        
        # Aggregate stat-arb signals from all pairs
        for key, value in signals.items():
            if key.startswith('stat_arb_') and isinstance(value, dict):
                signal = value.get('signal', 0.0)
                confidence = value.get('confidence', 0.0)
                is_cointegrated = value.get('is_cointegrated', False)
                
                if is_cointegrated and confidence > 0.5:
                    # Weight signal by confidence
                    weighted_signal = signal * confidence
                    total_bias += weighted_signal
                    signal_count += 1
        
        # Average the signals
        if signal_count > 0:
            avg_bias = total_bias / signal_count
        else:
            avg_bias = 0.0
        
        # Apply maximum bias limit
        return np.clip(avg_bias, -self.max_bias, self.max_bias)
    
    def _calculate_quote_sizes(self, position: float, volatility: float, 
                             signals: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate bid and ask sizes based on position, volatility, and signals
        """
        # Base size
        base_size = self.base_qty
        
        # Volatility scaling
        if self.vol_scale:
            # Reduce size in high volatility
            vol_factor = max(0.2, 1.0 / (1.0 + volatility * 50.0))
            base_size *= vol_factor
        
        # Position-dependent sizing
        # Reduce size on the side that would increase position
        bid_size = base_size
        ask_size = base_size
        
        if position > 0:
            # Long position - reduce bid size to discourage more buying
            bid_size *= max(0.3, 1.0 - abs(position) / 200.0)
        elif position < 0:
            # Short position - reduce ask size to discourage more selling
            ask_size *= max(0.3, 1.0 - abs(position) / 200.0)
        
        # Imbalance adjustment
        imbalance = signals.get('imbalance', 0.0)
        if imbalance > 0.1:
            # More bid volume - increase ask size
            ask_size *= 1.2
        elif imbalance < -0.1:
            # More ask volume - increase bid size
            bid_size *= 1.2
        
        # Apply leverage limits
        max_size = self.max_leverage * base_size
        bid_size = min(bid_size, max_size)
        ask_size = min(ask_size, max_size)
        
        # Minimum size
        min_size = 1.0
        bid_size = max(bid_size, min_size)
        ask_size = max(ask_size, min_size)
        
        return bid_size, ask_size
    
    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to tick size"""
        return round(price / tick_size) * tick_size
    
    def create_order_requests(self, quote: Quote, current_orders: List[Any]) -> List[OrderRequest]:
        """
        Create order requests from quote, considering current orders
        
        Returns list of order requests (cancels + new orders)
        """
        requests = []
        
        # Check if we need to update quotes
        needs_update = self._needs_quote_update(quote, current_orders)
        
        if needs_update:
            # Cancel all current orders first
            # (In practice, you might want to be more selective about cancels)
            
            # Create new bid order
            if quote.bid_size > 0:
                bid_request = OrderRequest(
                    side=Side.BUY,
                    quantity=quote.bid_size,
                    price=quote.bid_price,
                    order_type=OrderType.LIMIT,
                    client_order_id=f"bid_{quote.timestamp}"
                )
                requests.append(bid_request)
            
            # Create new ask order
            if quote.ask_size > 0:
                ask_request = OrderRequest(
                    side=Side.SELL,
                    quantity=quote.ask_size,
                    price=quote.ask_price,
                    order_type=OrderType.LIMIT,
                    client_order_id=f"ask_{quote.timestamp}"
                )
                requests.append(ask_request)
        
        return requests
    
    def _needs_quote_update(self, new_quote: Quote, current_orders: List[Any]) -> bool:
        """
        Determine if quotes need to be updated based on price/size changes
        """
        if not current_orders:
            return True  # No current orders, need to quote
        
        if self.last_quote is None:
            return True  # First quote
        
        # Check for significant price changes
        price_threshold = 0.001  # 0.1% price change threshold
        
        last_bid = self.last_quote.bid_price
        last_ask = self.last_quote.ask_price
        
        bid_change = abs(new_quote.bid_price - last_bid) / last_bid if last_bid > 0 else 1.0
        ask_change = abs(new_quote.ask_price - last_ask) / last_ask if last_ask > 0 else 1.0
        
        if bid_change > price_threshold or ask_change > price_threshold:
            return True
        
        # Check for significant size changes
        size_threshold = 0.2  # 20% size change threshold
        
        last_bid_size = self.last_quote.bid_size
        last_ask_size = self.last_quote.ask_size
        
        if last_bid_size > 0:
            bid_size_change = abs(new_quote.bid_size - last_bid_size) / last_bid_size
            if bid_size_change > size_threshold:
                return True
        
        if last_ask_size > 0:
            ask_size_change = abs(new_quote.ask_size - last_ask_size) / last_ask_size
            if ask_size_change > size_threshold:
                return True
        
        return False
    
    def get_quote_analytics(self) -> Dict[str, Any]:
        """Get analytics on quoting performance"""
        if not self.quote_history:
            return {}
        
        recent_quotes = self.quote_history[-100:]  # Last 100 quotes
        
        spreads = [q.ask_price - q.bid_price for q in recent_quotes]
        skews = [q.skew for q in recent_quotes]
        stat_arb_biases = [q.stat_arb_bias for q in recent_quotes]
        
        analytics = {
            'num_quotes': len(self.quote_history),
            'avg_spread': np.mean(spreads) if spreads else 0,
            'spread_std': np.std(spreads) if spreads else 0,
            'avg_skew': np.mean(skews) if skews else 0,
            'skew_std': np.std(skews) if skews else 0,
            'avg_stat_arb_bias': np.mean(stat_arb_biases) if stat_arb_biases else 0,
            'stat_arb_bias_std': np.std(stat_arb_biases) if stat_arb_biases else 0
        }
        
        if self.last_quote:
            analytics.update({
                'last_bid': self.last_quote.bid_price,
                'last_ask': self.last_quote.ask_price,
                'last_spread': self.last_quote.ask_price - self.last_quote.bid_price,
                'last_reservation_price': self.last_quote.reservation_price,
                'last_skew': self.last_quote.skew,
                'last_stat_arb_bias': self.last_quote.stat_arb_bias
            })
        
        return analytics
    
    def should_quote(self, market_data: Dict[str, Any], risk_check: Dict[str, Any]) -> bool:
        """
        Determine if we should be quoting based on market conditions and risk
        """
        # Don't quote during market halts or extreme conditions
        spread = market_data.get('spread')
        if spread and spread > 0.1:  # 10% spread is too wide
            return False
        
        # Don't quote if risk limits are breached
        if risk_check.get('kill_switch_active', False):
            return False
        
        if risk_check.get('position_limit_breach', False):
            return False
        
        # Don't quote if volatility is extremely high
        volatility = market_data.get('volatility', 0)
        if volatility > 0.1:  # 10% volatility
            return False
        
        return True


# Utility functions for quote optimization
def quotes_avellaneda_stoikov(mid: float, q: float, sigma: float, tau: float, 
                             gamma: float, k: float, phi: float, bias: float = 0.0) -> Tuple[float, float]:
    """
    Generate quotes using Avellaneda-Stoikov framework
    
    Args:
        mid: Fair value (mid price or microprice)
        q: Current inventory (signed)
        sigma: Volatility
        tau: Time to horizon (0-1)
        gamma: Risk aversion parameter
        k: Market depth parameter
        phi: Inventory penalty parameter
        bias: Statistical arbitrage bias (-1 to 1)
    
    Returns:
        (bid_price, ask_price)
    """
    # Reservation price
    r = mid - q * gamma * (sigma ** 2) * tau
    
    # Optimal half-spread
    if gamma > 0 and k > 0:
        delta = (1.0 / gamma) * np.log(1.0 + gamma / k) + 0.5 * gamma * (sigma ** 2) * tau
    else:
        delta = 0.01
    
    # Inventory skew
    skew = phi * q
    
    # Apply bias
    bias_adjustment = bias * delta
    
    # Final quotes
    bid = r - delta - skew - bias_adjustment
    ask = r + delta - skew - bias_adjustment
    
    return bid, ask


def optimal_spread_AS(gamma: float, k: float, sigma: float, tau: float) -> float:
    """Calculate optimal spread using Avellaneda-Stoikov formula"""
    if gamma <= 0 or k <= 0:
        return 0.01
    
    return 2.0 * ((1.0 / gamma) * np.log(1.0 + gamma / k) + 0.5 * gamma * (sigma ** 2) * tau)


def inventory_penalty(q: float, phi: float, max_q: float) -> float:
    """Calculate inventory penalty with position limits"""
    if max_q <= 0:
        return phi * q
    
    # Exponential penalty near limits
    q_normalized = q / max_q
    if abs(q_normalized) > 0.8:
        penalty_multiplier = np.exp(abs(q_normalized) - 0.8)
    else:
        penalty_multiplier = 1.0
    
    return phi * q * penalty_multiplier
