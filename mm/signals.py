"""Signal generation for microprice, volatility, and statistical arbitrage."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

def simple_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    if denominator == 0:
        return 0.0, y_mean
    
    slope = numerator / denominator
    intercept = y_mean - slope * X_mean
    
    return slope, intercept

# Check for optional dependencies
try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, using simplified cointegration tests")

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class VolatilitySignal:
    """Volatility signal with multiple estimators"""
    ewma_vol: float
    realized_vol: float
    parkinson_vol: float
    timestamp: float


@dataclass
class StatArbSignal:
    """Statistical arbitrage signal"""
    z_score: float
    spread: float
    hedge_ratio: float
    signal: float  # -1 to 1, with 0 being neutral
    confidence: float  # 0 to 1
    is_cointegrated: bool
    timestamp: float


class MicropriceCalculator:
    """Calculate microprice and order book imbalance signals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('microprice_alpha', 0.1)  # EWMA for microprice smoothing
        self.smoothed_microprice = None
        
    def calculate_microprice(self, bid: float, ask: float, bid_vol: float, ask_vol: float) -> float:
        """
        Calculate microprice: weighted average of bid/ask by opposite volume
        Formula: (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
        """
        if bid_vol + ask_vol == 0:
            return (bid + ask) / 2
        
        microprice = (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
        
        # Apply EWMA smoothing
        if self.smoothed_microprice is None:
            self.smoothed_microprice = microprice
        else:
            self.smoothed_microprice = (1 - self.alpha) * self.smoothed_microprice + self.alpha * microprice
        
        return self.smoothed_microprice
    
    def calculate_imbalance(self, bid_vol: float, ask_vol: float) -> float:
        """Calculate order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)"""
        if bid_vol + ask_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)


class VolatilityEstimator:
    """Multiple volatility estimators with EWMA"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('ewma_alpha', 0.2)
        self.lookback = config.get('vol_lookback', 100)
        self.min_vol = config.get('min_vol', 0.001)
        
        # Price history for different estimators
        self.prices = deque(maxlen=self.lookback)
        self.returns = deque(maxlen=self.lookback)
        self.high_prices = deque(maxlen=self.lookback)
        self.low_prices = deque(maxlen=self.lookback)
        
        # EWMA volatility state
        self.ewma_var = None
        
    def update(self, price: float, high: Optional[float] = None, low: Optional[float] = None) -> VolatilitySignal:
        """Update volatility estimates with new price"""
        self.prices.append(price)
        
        if high is not None:
            self.high_prices.append(high)
        else:
            self.high_prices.append(price)
            
        if low is not None:
            self.low_prices.append(low)
        else:
            self.low_prices.append(price)
        
        # Calculate return if we have previous price
        if len(self.prices) > 1:
            ret = np.log(price / self.prices[-2])
            self.returns.append(ret)
            
            # Update EWMA variance
            if self.ewma_var is None:
                self.ewma_var = ret ** 2
            else:
                self.ewma_var = (1 - self.alpha) * self.ewma_var + self.alpha * (ret ** 2)
        
        # Calculate different volatility estimates
        ewma_vol = max(np.sqrt(self.ewma_var) if self.ewma_var else 0, self.min_vol)
        
        realized_vol = self.min_vol
        if len(self.returns) > 10:
            realized_vol = max(np.std(list(self.returns)) * np.sqrt(252 * 24 * 60), self.min_vol)
        
        parkinson_vol = self.min_vol
        if len(self.high_prices) > 10:
            # Parkinson estimator: more efficient use of OHLC data
            hl_ratios = [np.log(h/l)**2 for h, l in zip(self.high_prices, self.low_prices) if h > 0 and l > 0]
            if hl_ratios:
                parkinson_var = np.mean(hl_ratios) / (4 * np.log(2))
                parkinson_vol = max(np.sqrt(parkinson_var * 252 * 24 * 60), self.min_vol)
        
        return VolatilitySignal(
            ewma_vol=ewma_vol,
            realized_vol=realized_vol,
            parkinson_vol=parkinson_vol,
            timestamp=len(self.prices)
        )
    
    def get_current_volatility(self) -> float:
        """Get current best volatility estimate"""
        if self.ewma_var is None:
            return self.min_vol
        return max(np.sqrt(self.ewma_var), self.min_vol)


class CointegrationAnalyzer:
    """Cointegration analysis for statistical arbitrage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback = config.get('lookback', 200)
        self.adf_pval_max = config.get('adf_pval_max', 0.05)
        self.z_enter = config.get('z_enter', 2.0)
        self.z_exit = config.get('z_exit', 0.5)
        
        # Price series storage
        self.y1_prices = deque(maxlen=self.lookback)
        self.y2_prices = deque(maxlen=self.lookback)
        
        # Cointegration state
        self.hedge_ratio = 1.0
        self.spread_mean = 0.0
        self.spread_std = 1.0
        self.is_cointegrated = False
        self.last_signal = 0.0
        
    def update(self, price1: float, price2: float) -> StatArbSignal:
        """Update with new price pair and return signal"""
        self.y1_prices.append(price1)
        self.y2_prices.append(price2)
        
        # Need minimum history for analysis
        if len(self.y1_prices) < min(50, self.lookback // 2):
            return StatArbSignal(
                z_score=0.0,
                spread=0.0,
                hedge_ratio=self.hedge_ratio,
                signal=0.0,
                confidence=0.0,
                is_cointegrated=False,
                timestamp=len(self.y1_prices)
            )
        
        # Perform cointegration test
        is_coint, hedge_ratio = self._test_cointegration(
            list(self.y1_prices), list(self.y2_prices)
        )
        
        if is_coint:
            self.hedge_ratio = hedge_ratio
            self.is_cointegrated = True
        
        # Calculate spread and z-score
        spread = price1 - self.hedge_ratio * price2
        z_score = self._calculate_zscore(spread)
        
        # Generate signal
        signal = self._generate_signal(z_score)
        
        # Calculate confidence based on cointegration strength
        confidence = 1.0 if is_coint else 0.5
        
        return StatArbSignal(
            z_score=z_score,
            spread=spread,
            hedge_ratio=self.hedge_ratio,
            signal=signal,
            confidence=confidence,
            is_cointegrated=is_coint,
            timestamp=len(self.y1_prices)
        )
    
    def _test_cointegration(self, y1: List[float], y2: List[float]) -> Tuple[bool, float]:
        """Test for cointegration using Engle-Granger method"""
        if len(y1) < 20 or len(y2) < 20:
            return False, 1.0
        
        try:
            # Convert to numpy arrays
            y1_arr = np.array(y1)
            y2_arr = np.array(y2)
            
            # Estimate hedge ratio using OLS
            hedge_ratio, _ = self._estimate_hedge_ratio(y1_arr, y2_arr)
            
            # Calculate residuals
            residuals = y1_arr - hedge_ratio * y2_arr
            
            # Test residuals for stationarity
            if HAS_STATSMODELS:
                try:
                    adf_stat, p_value, _, _, _, _ = adfuller(residuals, maxlag=None, regression='c')
                    is_cointegrated = p_value <= self.adf_pval_max
                except:
                    # Fallback to simple variance ratio test
                    is_cointegrated = self._simple_stationarity_test(residuals)
            else:
                # Simple stationarity test
                is_cointegrated = self._simple_stationarity_test(residuals)
            
            return is_cointegrated, hedge_ratio
            
        except Exception as e:
            return False, 1.0
    
    def _estimate_hedge_ratio(self, y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float]:
        """Estimate hedge ratio using linear regression"""
        if HAS_SKLEARN:
            # Use sklearn if available
            from sklearn.linear_model import LinearRegression
            X = y2.reshape(-1, 1)
            y = y1
            lr = LinearRegression().fit(X, y)
            hedge_ratio = lr.coef_[0]
            intercept = lr.intercept_
        else:
            # Use simple implementation
            hedge_ratio, intercept = simple_linear_regression(y2, y1)
        
        return hedge_ratio, intercept
    
    def _simple_stationarity_test(self, series: np.ndarray) -> bool:
        """Simple stationarity test using variance ratio"""
        if len(series) < 20:
            return False
        
        # Split series in half and compare variances
        mid = len(series) // 2
        var1 = np.var(series[:mid])
        var2 = np.var(series[mid:])
        
        # Also check for mean reversion
        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
        
        # Simple heuristics for stationarity
        variance_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 0
        
        return variance_ratio > 0.3 and autocorr < 0.7
    
    def _calculate_zscore(self, spread: float) -> float:
        """Calculate z-score of current spread"""
        if len(self.y1_prices) < 10:
            return 0.0
        
        # Calculate spread series
        spreads = []
        for i in range(len(self.y1_prices)):
            if i < len(self.y2_prices):
                s = self.y1_prices[i] - self.hedge_ratio * self.y2_prices[i]
                spreads.append(s)
        
        if len(spreads) < 5:
            return 0.0
        
        # Update running statistics
        self.spread_mean = np.mean(spreads)
        self.spread_std = np.std(spreads)
        
        if self.spread_std == 0:
            return 0.0
        
        z_score = (spread - self.spread_mean) / self.spread_std
        return z_score
    
    def _generate_signal(self, z_score: float) -> float:
        """Generate trading signal from z-score"""
        if not self.is_cointegrated:
            return 0.0
        
        # Mean reversion signal
        if abs(z_score) > self.z_enter:
            # Strong signal for mean reversion
            signal = -np.sign(z_score)  # Sell if z > threshold, buy if z < -threshold
            self.last_signal = signal
        elif abs(z_score) < self.z_exit:
            # Exit signal
            signal = 0.0
            self.last_signal = 0.0
        else:
            # Hold current signal
            signal = self.last_signal
        
        # Clamp signal to [-1, 1]
        return np.clip(signal, -1.0, 1.0)


class ArrivalRateModel:
    """Model for order arrival rates λ±(δ) = A · exp(−k·δ)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.A = config.get('A', 1.8)  # Base arrival intensity
        self.k = config.get('k', 1.2)  # Spread sensitivity
        
    def arrival_rate(self, spread: float, side: str = 'both') -> float:
        """Calculate arrival rate for given spread"""
        if side == 'both':
            return 2 * self.A * np.exp(-self.k * spread)
        else:
            return self.A * np.exp(-self.k * spread)
    
    def optimal_spread(self, gamma: float, sigma: float, tau: float) -> float:
        """Calculate optimal half-spread using closed-form solution"""
        # Closed-form optimal spread: δ* ≈ (1/γ)·ln(1 + γ/k) + (γ·σ²·τ)/2
        return (1.0 / gamma) * np.log(1.0 + gamma / self.k) + 0.5 * gamma * sigma**2 * tau


class SignalManager:
    """Centralized signal management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize signal calculators
        self.microprice_calc = MicropriceCalculator(config.get('microprice', {}))
        self.vol_estimator = VolatilityEstimator(config.get('vol', {}))
        self.arrival_model = ArrivalRateModel(config.get('arrival_model', {}))
        
        # Stat-arb for pairs
        self.stat_arb_enabled = config.get('stat_arb', {}).get('enabled', False)
        self.coint_analyzers: Dict[str, CointegrationAnalyzer] = {}
        
        # Signal history
        self.signal_history: List[Dict[str, Any]] = []
        
    def add_cointegration_pair(self, symbol1: str, symbol2: str):
        """Add cointegration pair for stat-arb"""
        if self.stat_arb_enabled:
            pair_key = f"{symbol1}_{symbol2}"
            self.coint_analyzers[pair_key] = CointegrationAnalyzer(self.config.get('stat_arb', {}))
    
    def update_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update all signals with new market data"""
        timestamp = market_data.get('timestamp', 0)
        signals = {'timestamp': timestamp}
        
        # Basic market data
        bid = market_data.get('bid')
        ask = market_data.get('ask')
        bid_vol = market_data.get('bid_size', 0)
        ask_vol = market_data.get('ask_size', 0)
        
        if bid and ask:
            # Microprice and imbalance
            microprice = self.microprice_calc.calculate_microprice(bid, ask, bid_vol, ask_vol)
            imbalance = self.microprice_calc.calculate_imbalance(bid_vol, ask_vol)
            
            signals['microprice'] = microprice
            signals['imbalance'] = imbalance
            signals['mid_price'] = (bid + ask) / 2
            signals['spread'] = ask - bid
            
            # Volatility
            mid_price = (bid + ask) / 2
            vol_signal = self.vol_estimator.update(mid_price)
            signals['volatility'] = vol_signal.ewma_vol
            signals['realized_vol'] = vol_signal.realized_vol
            
            # Arrival rates
            half_spread = (ask - bid) / 2
            signals['arrival_rate'] = self.arrival_model.arrival_rate(half_spread)
        
        # Statistical arbitrage signals
        if self.stat_arb_enabled and 'pairs_data' in market_data:
            pairs_data = market_data['pairs_data']
            for pair_key, coint_analyzer in self.coint_analyzers.items():
                if pair_key in pairs_data:
                    price1, price2 = pairs_data[pair_key]
                    stat_arb_signal = coint_analyzer.update(price1, price2)
                    signals[f'stat_arb_{pair_key}'] = {
                        'z_score': stat_arb_signal.z_score,
                        'signal': stat_arb_signal.signal,
                        'confidence': stat_arb_signal.confidence,
                        'hedge_ratio': stat_arb_signal.hedge_ratio,
                        'is_cointegrated': stat_arb_signal.is_cointegrated
                    }
        
        # Store signal history
        self.signal_history.append(signals.copy())
        
        # Keep only recent history
        max_history = 1000
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
        
        return signals
    
    def get_current_volatility(self) -> float:
        """Get current volatility estimate"""
        return self.vol_estimator.get_current_volatility()
    
    def get_stat_arb_signal(self, pair_key: str) -> Optional[float]:
        """Get current stat-arb signal for pair"""
        if pair_key in self.coint_analyzers:
            if self.signal_history:
                last_signals = self.signal_history[-1]
                stat_arb_key = f'stat_arb_{pair_key}'
                if stat_arb_key in last_signals:
                    return last_signals[stat_arb_key]['signal']
        return None
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signals"""
        if not self.signal_history:
            return {}
        
        latest = self.signal_history[-1]
        
        summary = {
            'timestamp': latest.get('timestamp'),
            'microprice': latest.get('microprice'),
            'volatility': latest.get('volatility'),
            'imbalance': latest.get('imbalance'),
            'spread': latest.get('spread')
        }
        
        # Add stat-arb signals
        for key, value in latest.items():
            if key.startswith('stat_arb_'):
                summary[key] = value
        
        return summary


# Utility functions for cointegration testing
def hedge_ratio(y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float]:
    """Calculate hedge ratio using linear regression"""
    if HAS_SKLEARN:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(y2.reshape(-1, 1), y1)
        return lr.coef_[0], lr.intercept_
    else:
        return simple_linear_regression(y2, y1)


def zscore(spread: np.ndarray, lookback: int = 200) -> float:
    """Calculate z-score of spread"""
    if len(spread) < lookback:
        s = spread
    else:
        s = spread[-lookback:]
    
    if len(s) < 2:
        return 0.0
    
    mean_spread = np.mean(s)
    std_spread = np.std(s)
    
    if std_spread == 0:
        return 0.0
    
    return (s[-1] - mean_spread) / std_spread


def eg_cointegration(y1: np.ndarray, y2: np.ndarray, pval_max: float = 0.05) -> Tuple[bool, float, np.ndarray]:
    """Engle-Granger cointegration test"""
    beta, alpha = hedge_ratio(y1, y2)
    resid = y1 - (alpha + beta * y2)
    
    if HAS_STATSMODELS:
        try:
            adf_stat, pval, _, _, _, _ = adfuller(resid)
            is_cointegrated = pval <= pval_max
        except:
            # Fallback
            is_cointegrated = np.std(resid) < 0.1 * np.std(y1)
    else:
        # Simple test: residuals should have lower variance than original series
        is_cointegrated = np.std(resid) < 0.1 * np.std(y1)
    
    return is_cointegrated, beta, resid


def statarb_signal(y1: np.ndarray, y2: np.ndarray, z_enter: float = 2.0, 
                  z_exit: float = 0.5, lookback: int = 200) -> float:
    """Generate stat-arb signal from price series"""
    if len(y1) < 20 or len(y2) < 20:
        return 0.0
    
    is_coint, beta, resid = eg_cointegration(y1, y2)
    
    if not is_coint or len(resid) < lookback:
        return 0.0
    
    z = zscore(resid, lookback)
    
    # Mean reversion signal
    if abs(z) > z_enter:
        return -np.sign(z)  # Fade the extremes
    elif abs(z) < z_exit:
        return 0.0  # Exit position
    else:
        return np.nan  # Hold current position
