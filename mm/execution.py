"""Order execution engine with position tracking and P&L calculation."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from collections import defaultdict

from .orderbook import OrderBook, Order, Side, Fill


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"  # Immediate or Cancel


class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderRequest:
    """Order request from strategy"""
    side: Side
    quantity: float
    price: Optional[float] = None
    order_type: OrderType = OrderType.LIMIT
    client_order_id: Optional[str] = None


@dataclass
class ExecutionOrder:
    """Internal order representation"""
    order_id: str
    client_order_id: str
    side: Side
    quantity: float
    price: Optional[float]
    order_type: OrderType
    status: OrderStatus
    timestamp: float
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0
    
    @property
    def remaining_qty(self) -> float:
        return self.quantity - self.filled_qty
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


@dataclass
class ExecutionReport:
    """Execution report for fills"""
    order_id: str
    client_order_id: str
    side: Side
    filled_qty: float
    fill_price: float
    timestamp: float
    fees: float
    rebates: float
    is_aggressive: bool
    cumulative_qty: float
    avg_price: float


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: float = 0.0
    vwap: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0
    
    def update_position(self, fill_qty: float, fill_price: float, fees: float = 0.0, rebates: float = 0.0):
        """Update position with new fill"""
        if self.quantity == 0:
            # New position
            self.quantity = fill_qty
            self.vwap = fill_price
        elif (self.quantity > 0 and fill_qty > 0) or (self.quantity < 0 and fill_qty < 0):
            # Adding to position
            total_notional = abs(self.quantity) * self.vwap + abs(fill_qty) * fill_price
            total_qty = abs(self.quantity) + abs(fill_qty)
            self.vwap = total_notional / total_qty
            self.quantity += fill_qty
        else:
            # Reducing or reversing position
            if abs(fill_qty) <= abs(self.quantity):
                # Reducing position - realize P&L
                if self.quantity > 0:  # Long position, selling
                    pnl_per_share = fill_price - self.vwap
                else:  # Short position, buying
                    pnl_per_share = self.vwap - fill_price
                
                self.realized_pnl += abs(fill_qty) * pnl_per_share
                self.quantity += fill_qty
            else:
                # Reversing position
                reduce_qty = -self.quantity
                new_qty = fill_qty + self.quantity
                
                # Realize P&L on the reduction
                if self.quantity > 0:
                    pnl_per_share = fill_price - self.vwap
                else:
                    pnl_per_share = self.vwap - fill_price
                
                self.realized_pnl += abs(reduce_qty) * pnl_per_share
                
                # New position in opposite direction
                self.quantity = new_qty
                self.vwap = fill_price
        
        self.fees_paid += fees
        self.rebates_earned += rebates
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        elif self.quantity > 0:
            self.unrealized_pnl = self.quantity * (current_price - self.vwap)
        else:
            self.unrealized_pnl = abs(self.quantity) * (self.vwap - current_price)
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid + self.rebates_earned


class ExecutionEngine:
    """Order execution engine with realistic market simulation"""
    
    def __init__(self, config: Dict[str, Any], orderbook: OrderBook):
        self.config = config
        self.orderbook = orderbook
        
        # Execution parameters
        self.latency_ms = config.get('latency_ms', 10)
        self.fees_bps = config.get('fees_bps', 0.4)
        self.rebate_bps = config.get('rebate_bps', 0.1)
        self.slippage_model = config.get('slippage_model', 'linear')
        
        # Order tracking
        self.orders: Dict[str, ExecutionOrder] = {}
        self.pending_orders: List[Tuple[float, ExecutionOrder]] = []  # (execution_time, order)
        
        # Position tracking
        self.positions: Dict[str, Position] = defaultdict(lambda: Position(""))
        
        # Risk limits
        self.max_position = config.get('max_position', 1000)
        self.max_orders_per_minute = config.get('max_orders_per_minute', 60)
        self.order_count_window: List[float] = []
        
        # Execution history
        self.execution_reports: List[ExecutionReport] = []
        
        # P&L tracking
        self.cash = config.get('start_cash', 100000.0)
        self.total_fees = 0.0
        self.total_rebates = 0.0
        
    def submit_order(self, request: OrderRequest, current_time: float) -> Optional[str]:
        """Submit order for execution"""
        # Check rate limits
        self._update_order_count(current_time)
        if len(self.order_count_window) >= self.max_orders_per_minute:
            return None  # Rate limited
        
        # Check position limits for new orders
        symbol = self.orderbook.symbol
        current_pos = self.positions[symbol].quantity
        
        if request.side == Side.BUY:
            new_pos = current_pos + request.quantity
        else:
            new_pos = current_pos - request.quantity
        
        if abs(new_pos) > self.max_position:
            return None  # Position limit
        
        # Create order
        order_id = str(uuid.uuid4())[:8]
        client_order_id = request.client_order_id or order_id
        
        order = ExecutionOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
            order_type=request.order_type,
            status=OrderStatus.PENDING,
            timestamp=current_time
        )
        
        self.orders[order_id] = order
        
        # Add latency delay
        execution_time = current_time + self.latency_ms / 1000.0
        self.pending_orders.append((execution_time, order))
        self.pending_orders.sort(key=lambda x: x[0])
        
        self.order_count_window.append(current_time)
        
        return order_id
    
    def cancel_order(self, order_id: str, current_time: float) -> bool:
        """Cancel order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.is_complete:
            return False
        
        # Remove from pending orders
        self.pending_orders = [(t, o) for t, o in self.pending_orders if o.order_id != order_id]
        
        # Cancel from orderbook if active
        if order.status == OrderStatus.ACTIVE:
            self.orderbook.cancel_order(f"our_{order_id}")
        
        order.status = OrderStatus.CANCELLED
        return True
    
    def step(self, current_time: float) -> List[ExecutionReport]:
        """Process pending orders and fills"""
        reports = []
        
        # Process pending orders that are ready for execution
        ready_orders = []
        remaining_orders = []
        
        for execution_time, order in self.pending_orders:
            if execution_time <= current_time:
                ready_orders.append(order)
            else:
                remaining_orders.append((execution_time, order))
        
        self.pending_orders = remaining_orders
        
        # Execute ready orders
        for order in ready_orders:
            if order.order_type == OrderType.MARKET:
                # Execute market order immediately
                reports.extend(self._execute_market_order(order, current_time))
            elif order.order_type == OrderType.IOC:
                # IOC order - try to fill immediately, cancel remainder
                reports.extend(self._execute_ioc_order(order, current_time))
            else:
                # Limit order - add to book
                self._add_limit_order_to_book(order, current_time)
        
        # Check for fills from market activity
        symbol = self.orderbook.symbol
        current_price = self.orderbook.get_mid_price()
        if current_price:
            self.positions[symbol].update_unrealized_pnl(current_price)
        
        # Simulate passive fills for our active orders based on proximity to touch
        reports.extend(self._simulate_passive_fills(current_time))

        return reports

    def _simulate_passive_fills(self, current_time: float) -> List[ExecutionReport]:
        """Simulate passive maker fills using a simple arrival rate model.

        If our quotes are at or within one tick of the current best prices,
        assume a Poisson fill process with higher intensity at the touch.
        """
        reports: List[ExecutionReport] = []
        best_bid = self.orderbook.get_best_bid()
        best_ask = self.orderbook.get_best_ask()
        if not (best_bid and best_ask):
            return reports

        best_bid_price, _ = best_bid
        best_ask_price, _ = best_ask

        tick_size = max(self.orderbook.tick_size, 0.01)
        base_lambda_at_touch = 12.0
        decay_per_tick = 0.08

        for order in list(self.orders.values()):
            if order.status != OrderStatus.ACTIVE or order.remaining_qty <= 0:
                continue

            # Determine how close our order is to best price
            if order.side == Side.BUY:
                ticks_from_touch = int(round((best_bid_price - (order.price or best_bid_price)) / tick_size))
            else:
                ticks_from_touch = int(round(((order.price or best_ask_price) - best_ask_price) / tick_size))

            # At touch or better (0 or negative means equal/better)
            if ticks_from_touch <= 0:
                lam = base_lambda_at_touch
            else:
                lam = max(0.0, base_lambda_at_touch * (1.0 - decay_per_tick * ticks_from_touch))

            if lam <= 0.0:
                continue

            step_seconds = max(self.latency_ms / 1000.0, 0.015)
            p_fill = min(0.25, 1.0 - np.exp(-lam * step_seconds))

            if np.random.random() < p_fill:
                fill_qty = min(order.remaining_qty, max(1.0, order.quantity * 1.0))
                fill_price = order.price or (best_ask_price if order.side == Side.BUY else best_bid_price)

                # Update order and portfolio
                rebates = self._calculate_rebates(fill_qty, fill_price)
                order.filled_qty += fill_qty
                if order.filled_qty > 0:
                    total_notional = order.avg_fill_price * (order.filled_qty - fill_qty) + fill_price * fill_qty
                    order.avg_fill_price = total_notional / order.filled_qty
                if order.filled_qty >= order.quantity:
                    order.status = OrderStatus.FILLED
                    # Remove from orderbook level
                    self.orderbook.cancel_order(f"our_{order.order_id}")

                # Position and cash
                symbol = self.orderbook.symbol
                pos_qty = fill_qty if order.side == Side.BUY else -fill_qty
                self.positions[symbol].update_position(pos_qty, fill_price, 0.0, rebates)
                if order.side == Side.BUY:
                    self.cash -= fill_qty * fill_price - rebates
                else:
                    self.cash += fill_qty * fill_price - rebates
                self.total_rebates += rebates

                # Report
                report = ExecutionReport(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    side=order.side,
                    filled_qty=fill_qty,
                    fill_price=fill_price,
                    timestamp=current_time,
                    fees=0.0,
                    rebates=rebates,
                    is_aggressive=False,
                    cumulative_qty=order.filled_qty,
                    avg_price=order.avg_fill_price
                )
                reports.append(report)
                self.execution_reports.append(report)

        return reports
    
    def _execute_market_order(self, order: ExecutionOrder, current_time: float) -> List[ExecutionReport]:
        """Execute market order with slippage"""
        reports = []
        
        # Get current market state
        if order.side == Side.BUY:
            best_ask = self.orderbook.get_best_ask()
            if not best_ask:
                order.status = OrderStatus.REJECTED
                return reports
            market_price = best_ask[0]
        else:
            best_bid = self.orderbook.get_best_bid()
            if not best_bid:
                order.status = OrderStatus.REJECTED
                return reports
            market_price = best_bid[0]
        
        # Apply slippage
        fill_price = self._apply_slippage(market_price, order.quantity, order.side)
        
        # Calculate fees (aggressive order pays fees)
        fees = self._calculate_fees(order.quantity, fill_price, is_aggressive=True)
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            filled_qty=order.quantity,
            fill_price=fill_price,
            timestamp=current_time,
            side=order.side,
            is_aggressive=True
        )
        
        # Update order
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.fees_paid = fees
        order.status = OrderStatus.FILLED
        
        # Update position
        symbol = self.orderbook.symbol
        pos_qty = order.quantity if order.side == Side.BUY else -order.quantity
        self.positions[symbol].update_position(pos_qty, fill_price, fees)
        
        # Update cash
        if order.side == Side.BUY:
            self.cash -= order.quantity * fill_price + fees
        else:
            self.cash += order.quantity * fill_price - fees
        
        self.total_fees += fees
        
        # Create execution report
        report = ExecutionReport(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            side=order.side,
            filled_qty=order.quantity,
            fill_price=fill_price,
            timestamp=current_time,
            fees=fees,
            rebates=0.0,
            is_aggressive=True,
            cumulative_qty=order.quantity,
            avg_price=fill_price
        )
        
        reports.append(report)
        self.execution_reports.append(report)
        
        return reports
    
    def _execute_ioc_order(self, order: ExecutionOrder, current_time: float) -> List[ExecutionReport]:
        """Execute IOC order"""
        reports = []
        
        # Try to fill against current market
        if order.side == Side.BUY:
            best_ask = self.orderbook.get_best_ask()
            if best_ask and order.price and order.price >= best_ask[0]:
                # Can fill at least partially
                fill_qty = min(order.quantity, best_ask[1])
                fill_price = best_ask[0]
                
                # Execute the fill
                fees = self._calculate_fees(fill_qty, fill_price, is_aggressive=True)
                
                order.filled_qty = fill_qty
                order.avg_fill_price = fill_price
                order.fees_paid = fees
                
                if fill_qty == order.quantity:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.CANCELLED  # Partial fill, remainder cancelled
                
                # Update position and cash
                symbol = self.orderbook.symbol
                self.positions[symbol].update_position(fill_qty, fill_price, fees)
                self.cash -= fill_qty * fill_price + fees
                self.total_fees += fees
                
                # Create report
                report = ExecutionReport(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    side=order.side,
                    filled_qty=fill_qty,
                    fill_price=fill_price,
                    timestamp=current_time,
                    fees=fees,
                    rebates=0.0,
                    is_aggressive=True,
                    cumulative_qty=fill_qty,
                    avg_price=fill_price
                )
                
                reports.append(report)
                self.execution_reports.append(report)
        
        else:  # SELL
            best_bid = self.orderbook.get_best_bid()
            if best_bid and order.price and order.price <= best_bid[0]:
                # Can fill at least partially
                fill_qty = min(order.quantity, best_bid[1])
                fill_price = best_bid[0]
                
                # Execute the fill
                fees = self._calculate_fees(fill_qty, fill_price, is_aggressive=True)
                
                order.filled_qty = fill_qty
                order.avg_fill_price = fill_price
                order.fees_paid = fees
                
                if fill_qty == order.quantity:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.CANCELLED
                
                # Update position and cash
                symbol = self.orderbook.symbol
                self.positions[symbol].update_position(-fill_qty, fill_price, fees)
                self.cash += fill_qty * fill_price - fees
                self.total_fees += fees
                
                # Create report
                report = ExecutionReport(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    side=order.side,
                    filled_qty=fill_qty,
                    fill_price=fill_price,
                    timestamp=current_time,
                    fees=fees,
                    rebates=0.0,
                    is_aggressive=True,
                    cumulative_qty=fill_qty,
                    avg_price=fill_price
                )
                
                reports.append(report)
                self.execution_reports.append(report)
        
        if not reports:
            # Couldn't fill, cancel
            order.status = OrderStatus.CANCELLED
        
        return reports
    
    def _add_limit_order_to_book(self, order: ExecutionOrder, current_time: float):
        """Add limit order to orderbook"""
        if not order.price:
            order.status = OrderStatus.REJECTED
            return
        
        # Create orderbook order with our prefix
        book_order = Order(
            order_id=f"our_{order.order_id}",
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            timestamp=current_time
        )
        
        success = self.orderbook.add_order(book_order)
        if success:
            order.status = OrderStatus.ACTIVE
        else:
            order.status = OrderStatus.REJECTED
    
    def process_fills(self, fills: List[Fill], current_time: float) -> List[ExecutionReport]:
        """Process fills from orderbook"""
        reports = []
        
        for fill in fills:
            if not fill.order_id.startswith("our_"):
                continue
            
            # Extract our order ID
            order_id = fill.order_id[4:]  # Remove "our_" prefix
            
            if order_id not in self.orders:
                continue
            
            order = self.orders[order_id]
            
            # Calculate fees/rebates (passive order earns rebates)
            rebates = self._calculate_rebates(fill.filled_qty, fill.fill_price)
            
            # Update order
            order.filled_qty += fill.filled_qty
            
            # Update average fill price
            if order.filled_qty > 0:
                total_notional = order.avg_fill_price * (order.filled_qty - fill.filled_qty) + fill.fill_price * fill.filled_qty
                order.avg_fill_price = total_notional / order.filled_qty
            
            order.rebates_earned += rebates
            
            if order.filled_qty >= order.quantity:
                order.status = OrderStatus.FILLED
                # Remove from orderbook
                self.orderbook.cancel_order(fill.order_id)
            
            # Update position
            symbol = self.orderbook.symbol
            pos_qty = fill.filled_qty if order.side == Side.BUY else -fill.filled_qty
            self.positions[symbol].update_position(pos_qty, fill.fill_price, 0.0, rebates)
            
            # Update cash
            if order.side == Side.BUY:
                self.cash -= fill.filled_qty * fill.fill_price - rebates
            else:
                self.cash += fill.filled_qty * fill.fill_price - rebates
            
            self.total_rebates += rebates
            
            # Create execution report
            report = ExecutionReport(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                side=order.side,
                filled_qty=fill.filled_qty,
                fill_price=fill.fill_price,
                timestamp=current_time,
                fees=0.0,
                rebates=rebates,
                is_aggressive=False,
                cumulative_qty=order.filled_qty,
                avg_price=order.avg_fill_price
            )
            
            reports.append(report)
            self.execution_reports.append(report)
        
        return reports
    
    def _apply_slippage(self, market_price: float, quantity: float, side: Side) -> float:
        """Apply slippage model to market orders"""
        if self.slippage_model == 'linear':
            # Linear slippage: price impact proportional to size
            impact_bps = min(quantity / 1000.0 * 5, 20)  # Max 20 bps impact
            impact = market_price * impact_bps / 10000.0
            
            if side == Side.BUY:
                return market_price + impact
            else:
                return market_price - impact
        
        return market_price
    
    def _calculate_fees(self, quantity: float, price: float, is_aggressive: bool) -> float:
        """Calculate trading fees"""
        if is_aggressive:
            return quantity * price * self.fees_bps / 10000.0
        return 0.0
    
    def _calculate_rebates(self, quantity: float, price: float) -> float:
        """Calculate rebates for passive orders"""
        return quantity * price * self.rebate_bps / 10000.0
    
    def _update_order_count(self, current_time: float):
        """Update order count window for rate limiting"""
        # Remove orders older than 1 minute
        cutoff_time = current_time - 60.0
        self.order_count_window = [t for t in self.order_count_window if t > cutoff_time]
    
    def get_position(self, symbol: str) -> Position:
        """Get current position for symbol"""
        return self.positions[symbol]
    
    def get_total_pnl(self) -> float:
        """Get total P&L across all positions"""
        total_pnl = 0.0
        for position in self.positions.values():
            total_pnl += position.total_pnl
        return total_pnl
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        return self.cash + self.get_total_pnl()
    
    def flatten_position(self, symbol: str, current_time: float) -> Optional[str]:
        """Flatten position with market order"""
        position = self.positions[symbol]
        
        if abs(position.quantity) < 0.001:  # Already flat
            return None
        
        # Create market order to flatten
        if position.quantity > 0:
            side = Side.SELL
            quantity = position.quantity
        else:
            side = Side.BUY
            quantity = abs(position.quantity)
        
        request = OrderRequest(
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        return self.submit_order(request, current_time)
    
    def get_active_orders(self) -> List[ExecutionOrder]:
        """Get all active orders"""
        return [order for order in self.orders.values() 
                if order.status == OrderStatus.ACTIVE]
    
    def cancel_all_orders(self, current_time: float) -> int:
        """Cancel all active orders"""
        cancelled_count = 0
        for order in self.get_active_orders():
            if self.cancel_order(order.order_id, current_time):
                cancelled_count += 1
        return cancelled_count
