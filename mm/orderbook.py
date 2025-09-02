"""Order book implementation with L2 data and queue modeling."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import bisect
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Individual order in the book"""
    order_id: str
    side: Side
    price: float
    quantity: float
    timestamp: float
    remaining_qty: float = field(init=False)
    
    def __post_init__(self):
        self.remaining_qty = self.quantity


@dataclass
class PriceLevel:
    """Price level containing orders at the same price"""
    price: float
    orders: List[Order] = field(default_factory=list)
    total_quantity: float = 0.0
    
    def add_order(self, order: Order):
        """Add order to this price level"""
        self.orders.append(order)
        self.total_quantity += order.remaining_qty
    
    def remove_order(self, order_id: str) -> bool:
        """Remove order from price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_quantity -= order.remaining_qty
                del self.orders[i]
                return True
        return False
    
    def fill_quantity(self, qty: float) -> List[Tuple[str, float]]:
        """Fill quantity from this level, return list of (order_id, filled_qty)"""
        fills = []
        remaining_to_fill = qty
        
        while remaining_to_fill > 0 and self.orders:
            order = self.orders[0]
            
            if order.remaining_qty <= remaining_to_fill:
                # Fully fill this order
                filled_qty = order.remaining_qty
                fills.append((order.order_id, filled_qty))
                remaining_to_fill -= filled_qty
                self.total_quantity -= filled_qty
                self.orders.pop(0)
            else:
                # Partially fill this order
                fills.append((order.order_id, remaining_to_fill))
                order.remaining_qty -= remaining_to_fill
                self.total_quantity -= remaining_to_fill
                remaining_to_fill = 0
        
        return fills


@dataclass
class Fill:
    """Order fill information"""
    order_id: str
    filled_qty: float
    fill_price: float
    timestamp: float
    side: Side
    is_aggressive: bool = False  # True if taker, False if maker


class OrderBook:
    """Level-2 orderbook with queue modeling"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Price levels - using sorted lists for efficiency
        self.bids: Dict[float, PriceLevel] = {}  # price -> PriceLevel
        self.asks: Dict[float, PriceLevel] = {}  # price -> PriceLevel
        
        # Sorted price lists for fast access
        self.bid_prices: List[float] = []  # Sorted descending
        self.ask_prices: List[float] = []  # Sorted ascending
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        
        # Market state
        self.last_trade_price: Optional[float] = None
        self.last_trade_time: Optional[float] = None
        
        # Queue position tracking for our orders
        self.our_orders: Dict[str, int] = {}  # order_id -> queue_position
        
    def round_price(self, price: float) -> float:
        """Round price to tick size"""
        return round(price / self.tick_size) * self.tick_size
    
    def add_order(self, order: Order) -> bool:
        """Add order to book"""
        if order.order_id in self.orders:
            return False
            
        price = self.round_price(order.price)
        order.price = price
        
        # Add to appropriate side
        if order.side == Side.BUY:
            if price not in self.bids:
                self.bids[price] = PriceLevel(price)
                bisect.insort(self.bid_prices, price)
                self.bid_prices.sort(reverse=True)  # Keep descending order
            
            self.bids[price].add_order(order)
            
        else:  # SELL
            if price not in self.asks:
                self.asks[price] = PriceLevel(price)
                bisect.insort(self.ask_prices, price)
                self.ask_prices.sort()  # Keep ascending order
            
            self.asks[price].add_order(order)
        
        self.orders[order.order_id] = order
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order from book"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        price = order.price
        
        # Remove from appropriate side
        if order.side == Side.BUY:
            if price in self.bids:
                success = self.bids[price].remove_order(order_id)
                if success and self.bids[price].total_quantity == 0:
                    del self.bids[price]
                    self.bid_prices.remove(price)
        else:
            if price in self.asks:
                success = self.asks[price].remove_order(order_id)
                if success and self.asks[price].total_quantity == 0:
                    del self.asks[price]
                    self.ask_prices.remove(price)
        
        del self.orders[order_id]
        if order_id in self.our_orders:
            del self.our_orders[order_id]
            
        return True
    
    def market_order(self, side: Side, quantity: float, timestamp: float) -> List[Fill]:
        """Execute market order against the book"""
        fills = []
        remaining_qty = quantity
        
        if side == Side.BUY:
            # Buy from asks (ascending price order)
            for price in self.ask_prices[:]:
                if remaining_qty <= 0:
                    break
                    
                level = self.asks[price]
                level_fills = level.fill_quantity(min(remaining_qty, level.total_quantity))
                
                for order_id, filled_qty in level_fills:
                    fills.append(Fill(
                        order_id=order_id,
                        filled_qty=filled_qty,
                        fill_price=price,
                        timestamp=timestamp,
                        side=Side.SELL,  # The passive side
                        is_aggressive=False
                    ))
                    remaining_qty -= filled_qty
                
                # Clean up empty level
                if level.total_quantity == 0:
                    del self.asks[price]
                    self.ask_prices.remove(price)
        
        else:  # SELL
            # Sell to bids (descending price order)
            for price in self.bid_prices[:]:
                if remaining_qty <= 0:
                    break
                    
                level = self.bids[price]
                level_fills = level.fill_quantity(min(remaining_qty, level.total_quantity))
                
                for order_id, filled_qty in level_fills:
                    fills.append(Fill(
                        order_id=order_id,
                        filled_qty=filled_qty,
                        fill_price=price,
                        timestamp=timestamp,
                        side=Side.BUY,  # The passive side
                        is_aggressive=False
                    ))
                    remaining_qty -= filled_qty
                
                # Clean up empty level
                if level.total_quantity == 0:
                    del self.bids[price]
                    self.bid_prices.remove(price)
        
        # Update last trade info
        if fills:
            self.last_trade_price = fills[-1].fill_price
            self.last_trade_time = timestamp
        
        return fills
    
    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (price, quantity)"""
        if not self.bid_prices:
            return None
        price = self.bid_prices[0]
        return price, self.bids[price].total_quantity
    
    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (price, quantity)"""
        if not self.ask_prices:
            return None
        price = self.ask_prices[0]
        return price, self.asks[price].total_quantity
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        elif best_bid:
            return best_bid[0]
        elif best_ask:
            return best_ask[0]
        else:
            return self.last_trade_price
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None
    
    def get_microprice(self) -> Optional[float]:
        """Calculate microprice: weighted average of best bid/ask by opposite volume"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if not (best_bid and best_ask):
            return self.get_mid_price()
        
        bid_price, bid_vol = best_bid
        ask_price, ask_vol = best_ask
        
        if bid_vol + ask_vol == 0:
            return (bid_price + ask_price) / 2
        
        # Microprice formula: (ask_price * bid_vol + bid_price * ask_vol) / (bid_vol + ask_vol)
        microprice = (ask_price * bid_vol + bid_price * ask_vol) / (bid_vol + ask_vol)
        return microprice
    
    def get_imbalance(self) -> float:
        """Calculate order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if not (best_bid and best_ask):
            return 0.0
        
        bid_vol = best_bid[1]
        ask_vol = best_ask[1]
        
        if bid_vol + ask_vol == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Get market depth for specified number of levels"""
        depth = {"bids": [], "asks": []}
        
        # Get bid levels (highest to lowest)
        for i, price in enumerate(self.bid_prices[:levels]):
            if price in self.bids:
                depth["bids"].append((price, self.bids[price].total_quantity))
        
        # Get ask levels (lowest to highest)
        for i, price in enumerate(self.ask_prices[:levels]):
            if price in self.asks:
                depth["asks"].append((price, self.asks[price].total_quantity))
        
        return depth
    
    def get_queue_position(self, order_id: str) -> Optional[int]:
        """Get queue position for our order (0 = front of queue)"""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        price = order.price
        
        if order.side == Side.BUY and price in self.bids:
            level = self.bids[price]
        elif order.side == Side.SELL and price in self.asks:
            level = self.asks[price]
        else:
            return None
        
        # Find position in queue
        for i, level_order in enumerate(level.orders):
            if level_order.order_id == order_id:
                return i
        
        return None
    
    def update_from_market_data(self, event) -> List[Fill]:
        """Update book from market data event"""
        fills = []
        
        if hasattr(event, 'bid') and hasattr(event, 'ask'):
            # Quote update - update synthetic market levels only, preserve our orders
            # Remove previous synthetic market orders
            for existing_order_id in list(self.orders.keys()):
                if existing_order_id.startswith("market_"):
                    self.cancel_order(existing_order_id)
            
            # Add new market best bid/ask levels
            if event.bid_size > 0:
                bid_order = Order(
                    order_id=f"market_bid_{event.timestamp}",
                    side=Side.BUY,
                    price=event.bid,
                    quantity=event.bid_size,
                    timestamp=event.timestamp
                )
                self.add_order(bid_order)
            
            if event.ask_size > 0:
                ask_order = Order(
                    order_id=f"market_ask_{event.timestamp}",
                    side=Side.SELL,
                    price=event.ask,
                    quantity=event.ask_size,
                    timestamp=event.timestamp
                )
                self.add_order(ask_order)
        
        elif hasattr(event, 'price') and hasattr(event, 'size'):
            # Trade event - execute against our orders if they would be filled
            trade_side = Side.BUY if event.side == "buy" else Side.SELL
            
            # Check if any of our orders would be filled
            if trade_side == Side.BUY:
                # Someone bought, check our asks
                for price in self.ask_prices:
                    if price <= event.price:
                        level = self.asks[price]
                        for order in level.orders[:]:
                            if order.order_id.startswith("our_"):
                                # Our order would be filled
                                fill_qty = min(order.remaining_qty, event.size)
                                fills.append(Fill(
                                    order_id=order.order_id,
                                    filled_qty=fill_qty,
                                    fill_price=price,
                                    timestamp=event.timestamp,
                                    side=Side.SELL,
                                    is_aggressive=False
                                ))
                                order.remaining_qty -= fill_qty
                                if order.remaining_qty <= 0:
                                    self.cancel_order(order.order_id)
            else:
                # Someone sold, check our bids
                for price in self.bid_prices:
                    if price >= event.price:
                        level = self.bids[price]
                        for order in level.orders[:]:
                            if order.order_id.startswith("our_"):
                                # Our order would be filled
                                fill_qty = min(order.remaining_qty, event.size)
                                fills.append(Fill(
                                    order_id=order.order_id,
                                    filled_qty=fill_qty,
                                    fill_price=price,
                                    timestamp=event.timestamp,
                                    side=Side.BUY,
                                    is_aggressive=False
                                ))
                                order.remaining_qty -= fill_qty
                                if order.remaining_qty <= 0:
                                    self.cancel_order(order.order_id)
            
            self.last_trade_price = event.price
            self.last_trade_time = event.timestamp
        
        return fills
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current book state for logging/analysis"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        return {
            'timestamp': self.last_trade_time,
            'symbol': self.symbol,
            'best_bid': best_bid[0] if best_bid else None,
            'best_ask': best_ask[0] if best_ask else None,
            'bid_size': best_bid[1] if best_bid else 0,
            'ask_size': best_ask[1] if best_ask else 0,
            'mid_price': self.get_mid_price(),
            'microprice': self.get_microprice(),
            'spread': self.get_spread(),
            'imbalance': self.get_imbalance(),
            'last_trade_price': self.last_trade_price,
            'total_orders': len(self.orders)
        }
