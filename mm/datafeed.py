"""Market data feed system."""

import numpy as np
import pandas as pd
from typing import Iterator, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import random


@dataclass
class MarketEvent:

    timestamp: float
    symbol: str
    event_type: str


@dataclass 
class QuoteEvent(MarketEvent):

    bid: float
    ask: float
    bid_size: float
    ask_size: float
    
    def __post_init__(self):
        self.event_type = "quote"


@dataclass
class TradeEvent(MarketEvent):

    price: float
    size: float
    side: str  # "buy" or "sell"
    
    def __post_init__(self):
        self.event_type = "trade"


@dataclass
class L2Event(MarketEvent):
    """Level 2 orderbook update"""
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
    
    def __post_init__(self):
        self.event_type = "l2_update"


class DataFeed(ABC):
    """Abstract base class for data feeds"""
    
    @abstractmethod
    def stream(self) -> Iterator[MarketEvent]:
        """Stream market events"""
        pass
    
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if data feed is complete"""
        pass


class HistoricalDataFeed(DataFeed):
    """Historical data replay from CSV/Parquet files"""
    
    def __init__(self, data_path: str, symbol: str = "ASSET1"):
        self.data_path = data_path
        self.symbol = symbol
        self.data = None
        self.current_idx = 0
        self._load_data()
    
    def _load_data(self):
        """Load historical data from file"""
        try:
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
                
            # Ensure timestamp column exists
            if 'timestamp' not in self.data.columns:
                self.data['timestamp'] = pd.date_range(
                    start='2024-07-01 09:30:00',
                    periods=len(self.data),
                    freq='1S'
                ).astype(int) // 10**9
                
        except FileNotFoundError:
            # Generate synthetic data if file doesn't exist
            print(f"File {self.data_path} not found, generating synthetic data")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_points: int = 10000):
        """Generate synthetic historical data for demo purposes"""
        timestamps = np.arange(n_points)
        
        # Generate price series with random walk
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, n_points)
        prices = 100.0 + np.cumsum(price_changes)
        
        # Generate bid/ask spreads
        spreads = np.random.exponential(0.01, n_points)
        bids = prices - spreads / 2
        asks = prices + spreads / 2
        
        # Generate sizes
        bid_sizes = np.random.exponential(100, n_points)
        ask_sizes = np.random.exponential(100, n_points)
        
        self.data = pd.DataFrame({
            'timestamp': timestamps,
            'bid': bids,
            'ask': asks,
            'bid_size': bid_sizes,
            'ask_size': ask_sizes
        })
    
    def stream(self) -> Iterator[MarketEvent]:
        """Stream historical market events"""
        while self.current_idx < len(self.data):
            row = self.data.iloc[self.current_idx]
            
            yield QuoteEvent(
                timestamp=float(row['timestamp']),
                symbol=self.symbol,
                event_type="quote",
                bid=float(row['bid']),
                ask=float(row['ask']),
                bid_size=float(row['bid_size']),
                ask_size=float(row['ask_size'])
            )
            
            self.current_idx += 1
    
    def is_complete(self) -> bool:
        return self.current_idx >= len(self.data)


class SimulatedDataFeed(DataFeed):
    """Simulated market data with configurable parameters"""
    
    def __init__(self, config: Dict[str, Any], symbol: str = "ASSET1"):
        self.config = config
        self.symbol = symbol
        self.current_time = 0.0
        self.current_price = config.get('initial_price', 100.0)
        self.event_count = 0
        self.max_events = config.get('max_events', 10000)
        
        # OU process parameters
        self.ou_theta = config.get('ou_theta', 0.1)
        self.ou_sigma = config.get('ou_sigma', 0.01)
        self.drift = config.get('drift', 0.0)
        
        # Arrival model parameters
        self.arrival_A = config.get('arrival_A', 1.8)
        self.arrival_k = config.get('arrival_k', 1.2)
        
        # Initialize random state
        np.random.seed(config.get('seed', 42))
        
    def _update_price(self, dt: float = 0.001) -> float:
        """Update price using Ornstein-Uhlenbeck process"""
        # OU process: dX = θ(μ - X)dt + σdW
        mean_price = self.config.get('initial_price', 100.0)
        
        drift_term = self.ou_theta * (mean_price - self.current_price) * dt
        diffusion_term = self.ou_sigma * np.sqrt(dt) * np.random.normal()
        
        self.current_price += drift_term + diffusion_term + self.drift * dt
        return self.current_price
    
    def _generate_quote_event(self) -> QuoteEvent:
        """Generate a quote event with realistic bid/ask spreads"""
        # Update price
        self._update_price()
        
        # Generate spread (wider during volatile periods)
        base_spread = self.config.get('min_spread', 0.01)
        vol_factor = max(1.0, abs(np.random.normal(0, 0.1)))
        spread = base_spread * vol_factor
        
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        # Generate sizes with exponential distribution
        bid_size = np.random.exponential(100)
        ask_size = np.random.exponential(100)
        
        return QuoteEvent(
            timestamp=self.current_time,
            symbol=self.symbol,
            event_type="quote",  # Add missing event_type
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size
        )
    
    def _generate_trade_event(self) -> TradeEvent:
        """Generate a trade event"""
        # Random side
        side = "buy" if np.random.random() > 0.5 else "sell"
        
        # Price near current level with some noise
        price_noise = np.random.normal(0, 0.001)
        price = self.current_price + price_noise
        
        # Size with exponential distribution
        size = np.random.exponential(50)
        
        return TradeEvent(
            timestamp=self.current_time,
            symbol=self.symbol,
            event_type="trade",  # Add missing event_type
            price=price,
            size=size,
            side=side
        )
    
    def stream(self) -> Iterator[MarketEvent]:
        """Stream simulated market events"""
        while self.event_count < self.max_events:
            # Poisson arrivals for events
            dt = np.random.exponential(0.1)  # Average 10 events per second
            self.current_time += dt
            
            # 80% quotes, 20% trades
            if np.random.random() < 0.8:
                yield self._generate_quote_event()
            else:
                yield self._generate_trade_event()
                
            self.event_count += 1
    
    def is_complete(self) -> bool:
        return self.event_count >= self.max_events


class MultiAssetDataFeed(DataFeed):
    """Data feed for multiple assets (for stat-arb pairs)"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        self.symbols = symbols
        self.config = config
        self.feeds = {}
        self.current_events = {}
        
        # Create individual feeds for each symbol
        for symbol in symbols:
            if config.get('mode') == 'historical':
                data_path = f"{config.get('data_dir', 'data')}/{symbol}.csv"
                self.feeds[symbol] = HistoricalDataFeed(data_path, symbol)
            else:
                # Add correlation for stat-arb
                symbol_config = config.copy()
                if len(symbols) > 1 and symbol == symbols[1]:
                    # Second asset correlated with first
                    symbol_config['initial_price'] = config.get('initial_price', 100.0) * 0.95
                
                self.feeds[symbol] = SimulatedDataFeed(symbol_config, symbol)
    
    def stream(self) -> Iterator[MarketEvent]:
        """Stream events from all assets in time order"""
        # Initialize iterators
        iterators = {symbol: iter(feed.stream()) for symbol, feed in self.feeds.items()}
        
        # Get first event from each feed
        for symbol, iterator in iterators.items():
            try:
                self.current_events[symbol] = next(iterator)
            except StopIteration:
                pass
        
        while self.current_events:
            # Find earliest event
            earliest_symbol = min(self.current_events.keys(), 
                                key=lambda s: self.current_events[s].timestamp)
            
            event = self.current_events[earliest_symbol]
            yield event
            
            # Get next event from this feed
            try:
                self.current_events[earliest_symbol] = next(iterators[earliest_symbol])
            except StopIteration:
                # This feed is exhausted
                del self.current_events[earliest_symbol]
    
    def is_complete(self) -> bool:
        return all(feed.is_complete() for feed in self.feeds.values())


def create_datafeed(config: Dict[str, Any]) -> DataFeed:
    """Factory function to create appropriate data feed"""
    mode = config.get('mode', 'sim')
    symbols = config.get('symbols', ['ASSET1'])
    
    if len(symbols) == 1:
        if mode == 'historical':
            data_path = f"{config.get('data_dir', 'data')}/{symbols[0]}.csv"
            return HistoricalDataFeed(data_path, symbols[0])
        else:
            return SimulatedDataFeed(config.get('market_data', {}), symbols[0])
    else:
        return MultiAssetDataFeed(symbols, config)
