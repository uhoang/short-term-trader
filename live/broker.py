"""Alpaca broker client for paper and live trading."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import structlog
from dotenv import load_dotenv

from signals.base import Direction, SignalEvent
from signals.strategy import TradeParams

load_dotenv()
logger = structlog.get_logger(__name__)


class AlpacaBroker:
    """Unified broker interface for Alpaca paper and live trading."""

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self._client = self._connect()

    def _connect(self) -> object:
        """Connect to Alpaca API."""
        from alpaca.trading.client import TradingClient

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        if not api_key or api_key.startswith("your_"):
            logger.warning("alpaca_api_key_not_configured")

        client = TradingClient(api_key, secret_key, paper=self.paper)
        logger.info("alpaca_connected", paper=self.paper)
        return client

    def get_account(self) -> dict[str, object]:
        """Get account info: equity, buying power, etc."""
        account = self._client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
        }

    def get_positions(self) -> list[dict[str, object]]:
        """Get all open positions."""
        positions = self._client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value,
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
            }
            for p in positions
        ]

    def get_latest_price(self, ticker: str) -> float | None:
        """Get latest quote/trade price for a ticker."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            data_client = StockHistoricalDataClient(
                os.getenv("ALPACA_API_KEY", ""),
                os.getenv("ALPACA_SECRET_KEY", ""),
            )
            request = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trades = data_client.get_stock_latest_trade(request)
            if ticker in trades:
                return float(trades[ticker].price)
        except Exception:
            logger.warning("latest_price_fetch_failed", ticker=ticker)
        return None

    def submit_order(
        self,
        signal: SignalEvent,
        position_value: float,
    ) -> str | None:
        """Submit a market order (buys) or limit order based on a signal.

        For LONG: submits a BUY market order (simplest, avoids stale limit issues).
        For SHORT: skipped unless short selling is explicitly enabled.

        Returns order ID on success, None on failure.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        # Skip short orders — most paper accounts don't support short selling
        if signal.direction == Direction.SHORT:
            logger.info(
                "short_order_skipped",
                ticker=signal.ticker,
                reason="Short selling disabled in paper mode",
            )
            return None

        # Use latest market price for sizing, not stale feature-store price
        current_price = self.get_latest_price(signal.ticker)
        if current_price is None:
            trade_params = signal.metadata.get("trade_params")
            current_price = trade_params.entry_price if trade_params else 0

        if current_price <= 0:
            logger.error("invalid_price", ticker=signal.ticker)
            return None

        qty = int(position_value / current_price)
        if qty <= 0:
            logger.warning(
                "position_too_small",
                ticker=signal.ticker,
                value=position_value,
                price=current_price,
            )
            return None

        request = MarketOrderRequest(
            symbol=signal.ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        logger.info(
            "submitting_order",
            ticker=signal.ticker,
            qty=qty,
            price=current_price,
            request=str(request),
        )

        try:
            order = self._client.submit_order(request)
            logger.info(
                "order_submitted",
                ticker=signal.ticker,
                side="buy",
                qty=qty,
                est_price=current_price,
                order_id=order.id,
            )
            return str(order.id)
        except Exception as e:
            # Extract Alpaca API error details
            error_msg = str(e)
            if hasattr(e, "status_code"):
                error_msg = f"HTTP {e.status_code}: {e}"
            if hasattr(e, "_error"):
                error_msg = f"{e._error}"
            logger.error(
                "order_submission_failed",
                ticker=signal.ticker,
                qty=qty,
                price=current_price,
                error=error_msg,
            )
            return None

    def submit_bracket_order(
        self,
        signal: SignalEvent,
        position_value: float,
    ) -> str | None:
        """Submit bracket order: entry + stop loss + take profit."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        trade_params = signal.metadata.get("trade_params")
        if not trade_params or not isinstance(trade_params, TradeParams):
            return None

        entry_price = trade_params.entry_price
        qty = int(position_value / entry_price)
        if qty <= 0:
            return None

        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL

        # Calculate stop and target
        if signal.direction == Direction.LONG:
            stop_price = round(entry_price * (1 - trade_params.stop_loss_pct), 2)
            take_profit = round(entry_price * (1 + trade_params.take_profit_pct), 2)
        else:
            stop_price = round(entry_price * (1 + trade_params.stop_loss_pct), 2)
            take_profit = round(entry_price * (1 - trade_params.take_profit_pct), 2)

        try:
            order = self._client.submit_order(
                LimitOrderRequest(
                    symbol=signal.ticker,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(entry_price, 2),
                    order_class="bracket",
                    stop_loss={"stop_price": stop_price},
                    take_profit={"limit_price": take_profit},
                )
            )
            logger.info(
                "bracket_order_submitted",
                ticker=signal.ticker,
                entry=entry_price,
                stop=stop_price,
                target=take_profit,
                order_id=order.id,
            )
            return str(order.id)
        except Exception:
            logger.exception("bracket_order_failed", ticker=signal.ticker)
            return None

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        try:
            cancelled = self._client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            logger.info("all_orders_cancelled", count=count)
            return count
        except Exception:
            logger.exception("cancel_all_failed")
            return 0

    def close_all_positions(self) -> int:
        """Close all open positions (emergency shutdown). Returns count closed."""
        try:
            closed = self._client.close_all_positions(cancel_orders=True)
            count = len(closed) if closed else 0
            logger.info("all_positions_closed", count=count)
            return count
        except Exception:
            logger.exception("close_all_failed")
            return 0

    def get_order_history(self, days: int = 7) -> list[dict[str, object]]:
        """Get recent order history for reconciliation."""
        from alpaca.trading.requests import GetOrdersRequest

        try:
            request = GetOrdersRequest(
                status="all",
                after=datetime.now() - timedelta(days=days),
            )
            orders = self._client.get_orders(request)
            return [
                {
                    "id": str(o.id),
                    "ticker": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.qty) if o.qty else 0,
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                    "status": o.status.value,
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception:
            logger.exception("order_history_failed")
            return []
