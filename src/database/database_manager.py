"""
Trading Bot Database Manager - Supabase Integration
====================================================
Captures comprehensive trade metrics, signals, and performance data
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from supabase import create_client, Client
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


class TradingDatabaseManager:
    """
    Manages all database operations for the trading bot
    Stores trades, signals, performance metrics, and system events
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase connection

        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase API key (service role key for server-side)
        """
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            logger.info("[DB] ✓ Connected to Supabase")

            self._last_signals: Dict[str, Dict] = {}

            # Test connection
            self._test_connection()

        except Exception as e:
            logger.error(f"[DB] Failed to connect to Supabase: {e}")
            raise

    def _test_connection(self):
        """Test database connection"""
        try:
            # Try a simple query
            result = self.supabase.table("trades").select("id").limit(1).execute()
            logger.info("[DB] ✓ Connection test passed")
        except Exception as e:
            logger.warning(f"[DB] Connection test: {e}")

    # ========================================================================
    # TRADE OPERATIONS
    # ========================================================================

    def _check_existing_trade(
        self,
        mt5_ticket: Optional[int] = None,
        binance_order_id: Optional[int] = None,
        position_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Check if a trade with the same identifier already exists"""
        try:
            if mt5_ticket is not None:
                result = (
                    self.supabase.table("trades")
                    .select("*")
                    .eq("mt5_ticket", mt5_ticket)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    logger.warning(
                        f"[DB] Trade with MT5 ticket {mt5_ticket} already exists (ID={result.data[0]['id']})"
                    )
                    return result.data[0]

            if binance_order_id is not None:
                result = (
                    self.supabase.table("trades")
                    .select("*")
                    .eq("binance_order_id", binance_order_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    logger.warning(
                        f"[DB] Trade with Binance order ID {binance_order_id} already exists (ID={result.data[0]['id']})"
                    )
                    return result.data[0]

            if position_id is not None:
                result = (
                    self.supabase.table("trades")
                    .select("*")
                    .eq("position_id", position_id)
                    .execute()
                )
                if result.data and len(result.data) > 0:
                    logger.warning(
                        f"[DB] Trade with position ID {position_id} already exists (ID={result.data[0]['id']})"
                    )
                    return result.data[0]

            return None

        except Exception as e:
            logger.error(f"[DB] Error checking existing trade: {e}")
            return None

    def insert_trade_entry(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        position_size_usd: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_id: Optional[str] = None,
        exchange: str = "binance",
        strategy: Optional[str] = None,
        signal_quality: Optional[float] = None,
        regime: Optional[str] = None,
        confidence_score: Optional[float] = None,
        mt5_ticket: Optional[int] = None,
        binance_order_id: Optional[int] = None,
        vtm_enabled: bool = False,
        metadata: Optional[Dict] = None,
        update_if_exists: bool = True,
    ) -> Tuple[Optional[int], bool]:
        """
        Record a new trade entry with duplicate detection

        Returns:
            Tuple of (Trade ID, is_new_trade)
        """
        try:
            existing_trade = self._check_existing_trade(
                mt5_ticket=mt5_ticket,
                binance_order_id=binance_order_id,
                position_id=position_id,
            )

            if existing_trade:
                if update_if_exists and existing_trade["status"] == "open":
                    trade_id = existing_trade["id"]
                    update_data = {
                        "entry_price": float(entry_price),
                        "quantity": float(quantity),
                        "position_size_usd": float(position_size_usd),
                        "stop_loss": float(stop_loss) if stop_loss else None,
                        "take_profit": float(take_profit) if take_profit else None,
                        "signal_quality": (
                            float(signal_quality) if signal_quality else None
                        ),
                        "confidence_score": (
                            float(confidence_score) if confidence_score else None
                        ),
                        "vtm_enabled": vtm_enabled,
                    }

                    if metadata:
                        existing_meta = (
                            json.loads(existing_trade["metadata"])
                            if existing_trade.get("metadata")
                            else {}
                        )
                        existing_meta.update(metadata)
                        update_data["metadata"] = self._serialize_safely(existing_meta)

                    result = (
                        self.supabase.table("trades")
                        .update(update_data)
                        .eq("id", trade_id)
                        .execute()
                    )

                    if result.data:
                        logger.info(
                            f"[DB] ✓ Trade entry updated: ID={trade_id}, {asset} {side.upper()}"
                        )
                        return trade_id, False
                else:
                    logger.info(
                        f"[DB] Trade already exists (ID={existing_trade['id']}, status={existing_trade['status']}), skipping insertion"
                    )
                    return existing_trade["id"], False

            trade_data = {
                "asset": asset,
                "symbol": symbol,
                "side": side,
                "entry_price": float(entry_price),
                "quantity": float(quantity),
                "position_size_usd": float(position_size_usd),
                "stop_loss": float(stop_loss) if stop_loss else None,
                "take_profit": float(take_profit) if take_profit else None,
                "position_id": position_id,
                "exchange": exchange,
                "strategy": strategy,
                "signal_quality": float(signal_quality) if signal_quality else None,
                "regime": regime,
                "confidence_score": (
                    float(confidence_score) if confidence_score else None
                ),
                "mt5_ticket": mt5_ticket,
                "binance_order_id": binance_order_id,
                "vtm_enabled": vtm_enabled,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "status": "open",
                "metadata": self._serialize_safely(metadata) if metadata else None,
            }

            result = self.supabase.table("trades").insert(trade_data).execute()

            if result.data and len(result.data) > 0:
                trade_id = result.data[0]["id"]
                logger.info(
                    f"[DB] ✓ Trade entry recorded: ID={trade_id}, {asset} {side.upper()}"
                )
                return trade_id, True
            else:
                logger.error("[DB] Failed to insert trade entry")
                return None, False

        except Exception as e:
            logger.error(f"[DB] Error inserting trade entry: {e}")
            return None, False

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        holding_time_hours: Optional[float] = None,
        final_quantity: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update trade with exit information"""
        try:
            update_data = {
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "pnl": float(pnl),
                "pnl_pct": float(pnl_pct),
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "status": "closed",
                "holding_time_hours": (
                    float(holding_time_hours) if holding_time_hours else None
                ),
                "final_quantity": float(final_quantity) if final_quantity else None,
            }

            if metadata:
                existing = (
                    self.supabase.table("trades")
                    .select("metadata")
                    .eq("id", trade_id)
                    .execute()
                )
                if existing.data and existing.data[0].get("metadata"):
                    existing_meta = json.loads(existing.data[0]["metadata"])
                    existing_meta.update(metadata)
                    update_data["metadata"] = self._serialize_safely(existing_meta)
                else:
                    update_data["metadata"] = self._serialize_safely(metadata)

            result = (
                self.supabase.table("trades")
                .update(update_data)
                .eq("id", trade_id)
                .execute()
            )

            if result.data:
                logger.info(
                    f"[DB] ✓ Trade exit recorded: ID={trade_id}, "
                    f"P&L=${pnl:.2f} ({pnl_pct:+.2f}%), Reason={exit_reason}"
                )
                return True
            return False

        except Exception as e:
            logger.error(f"[DB] Error updating trade exit: {e}")
            return False

    def update_trade_vtm_event(
        self,
        trade_id: int,
        event_type: str,
        old_value: Optional[float] = None,
        new_value: Optional[float] = None,
        current_price: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Record VTM (Veteran Trade Manager) events"""
        try:
            event_data = {
                "trade_id": trade_id,
                "event_type": event_type,
                "old_value": float(old_value) if old_value else None,
                "new_value": float(new_value) if new_value else None,
                "current_price": float(current_price) if current_price else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": self._serialize_safely(metadata) if metadata else None,
            }

            result = self.supabase.table("vtm_events").insert(event_data).execute()

            if result.data:
                logger.debug(f"[DB] VTM event recorded: {event_type}")
                return True
            return False

        except Exception as e:
            logger.error(f"[DB] Error recording VTM event: {e}")
            return False

    # ========================================================================
    # SIGNAL TRACKING
    # ========================================================================

    def _check_recent_signal(
        self, asset: str, signal: int, time_window_minutes: int = 5
    ) -> Optional[Dict]:
        """
        Check if a similar signal was recently recorded to avoid duplicates

        Args:
            asset: Asset name
            signal: Signal value (-1, 0, 1)
            time_window_minutes: Time window to check for duplicates

        Returns:
            Recent signal if found, None otherwise
        """
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (
                time_window_minutes * 60
            )
            cutoff_iso = datetime.fromtimestamp(cutoff_time, timezone.utc).isoformat()

            result = (
                self.supabase.table("signals")
                .select("*")
                .eq("asset", asset)
                .eq("signal", signal)
                .gte("timestamp", cutoff_iso)
                .execute()
            )

            if result.data and len(result.data) > 0:
                logger.debug(
                    f"[DB] Similar signal for {asset} found within {time_window_minutes} minutes"
                )
                return result.data[0]

            return None

        except Exception as e:
            logger.error(f"[DB] Error checking recent signal: {e}")
            return None

    def insert_signal_smart(
        self,
        asset: str,
        signal: int,
        signal_quality: float,
        regime: str,
        regime_confidence: float,
        mr_signal: int,
        mr_confidence: float,
        tf_signal: int,
        tf_confidence: float,
        ema_signal: Optional[int] = None,
        ema_confidence: Optional[float] = None,
        buy_score: Optional[float] = None,
        sell_score: Optional[float] = None,
        reasoning: Optional[str] = None,
        price: Optional[float] = None,
        ai_validated: bool = False,
        ai_modified: bool = False,
        ai_details: Optional[Dict] = None,
        executed: bool = False,
        metadata: Optional[Dict] = None,
        force_insert: bool = False,
    ) -> Tuple[Optional[int], bool]:
        """
        Smart signal insertion - only records when signal actually changes

        Args:
            force_insert: If True, bypass change detection and insert anyway

        Returns:
            Tuple of (Signal ID, was_inserted)
            - Signal ID: database primary key (or None if skipped)
            - was_inserted: True if new signal was inserted, False if skipped
        """
        try:
            # Prepare signal data
            signal_data = {
                "asset": asset,
                "signal": signal,
                "signal_quality": float(signal_quality),
                "regime": regime,
                "regime_confidence": float(regime_confidence),
                "mr_signal": mr_signal,
                "mr_confidence": float(mr_confidence),
                "tf_signal": tf_signal,
                "tf_confidence": float(tf_confidence),
                "ema_signal": ema_signal,
                "ema_confidence": float(ema_confidence) if ema_confidence else None,
                "buy_score": float(buy_score) if buy_score else None,
                "sell_score": float(sell_score) if sell_score else None,
                "reasoning": reasoning,
                "price": float(price) if price else None,
                "ai_validated": ai_validated,
                "ai_modified": ai_modified,
                "ai_details": (
                    self._serialize_safely(ai_details) if ai_details else None
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": self._serialize_safely(metadata) if metadata else None,
            }

            # Check if signal has changed (unless force_insert is True)
            if not force_insert and not self._has_signal_changed(asset, signal_data):
                return None, False

            # Insert new signal
            result = self.supabase.table("signals").insert(signal_data).execute()

            if result.data and len(result.data) > 0:
                signal_id = result.data[0]["id"]

                # Update cache with new signal
                self._last_signals[asset] = signal_data
                self._last_signals[asset]["id"] = signal_id

                logger.info(
                    f"[DB] ✓ Signal recorded: ID={signal_id}, {asset} signal={signal}, regime={regime}"
                )
                return signal_id, True

            return None, False

        except Exception as e:
            logger.error(f"[DB] Error inserting signal: {e}")
            return None, False

    def update_signal_execution(
        self, signal_id: int, executed: bool, trade_id: Optional[int] = None
    ) -> bool:
        """Mark signal as executed and link to trade"""
        try:
            update_data = {
                "executed": executed,
                "trade_id": trade_id,
                "execution_time": datetime.now(timezone.utc).isoformat(),
            }

            result = (
                self.supabase.table("signals")
                .update(update_data)
                .eq("id", signal_id)
                .execute()
            )
            return bool(result.data)

        except Exception as e:
            logger.error(f"[DB] Error updating signal execution: {e}")
            return False

    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================

    def insert_daily_summary(
        self,
        date: datetime,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        total_pnl_pct: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        profit_factor: float,
        max_drawdown: float,
        starting_capital: float,
        ending_capital: float,
        btc_trades: int = 0,
        gold_trades: int = 0,
        metadata: Optional[Dict] = None,
        update_if_exists: bool = True,
    ) -> bool:
        """Record daily performance summary with duplicate handling"""
        try:
            date_str = date.date().isoformat()

            existing = (
                self.supabase.table("daily_summaries")
                .select("*")
                .eq("date", date_str)
                .execute()
            )

            summary_data = {
                "date": date_str,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": float(total_pnl),
                "total_pnl_pct": float(total_pnl_pct),
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": float(profit_factor),
                "max_drawdown": float(max_drawdown),
                "starting_capital": float(starting_capital),
                "ending_capital": float(ending_capital),
                "btc_trades": btc_trades,
                "gold_trades": gold_trades,
                "metadata": self._serialize_safely(metadata) if metadata else None,
            }

            if existing.data and len(existing.data) > 0:
                if update_if_exists:
                    result = (
                        self.supabase.table("daily_summaries")
                        .update(summary_data)
                        .eq("date", date_str)
                        .execute()
                    )
                    if result.data:
                        logger.info(
                            f"[DB] ✓ Daily summary updated: {date_str}, P&L=${total_pnl:.2f}"
                        )
                        return True
                else:
                    logger.info(
                        f"[DB] Daily summary for {date_str} already exists, skipping"
                    )
                    return True
            else:
                result = (
                    self.supabase.table("daily_summaries")
                    .insert(summary_data)
                    .execute()
                )

                if result.data:
                    logger.info(
                        f"[DB] ✓ Daily summary recorded: {date_str}, P&L=${total_pnl:.2f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"[DB] Error inserting daily summary: {e}")
            return False

    def insert_portfolio_snapshot(
        self,
        total_value: float,
        cash: float,
        equity: float,
        total_exposure: float,
        open_positions: int,
        unrealized_pnl: float,
        realized_pnl_today: float,
        positions_detail: Optional[Dict] = None,
    ) -> bool:
        """Record portfolio state snapshot"""
        try:
            snapshot_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_value": float(total_value),
                "cash": float(cash),
                "equity": float(equity),
                "total_exposure": float(total_exposure),
                "open_positions": open_positions,
                "unrealized_pnl": float(unrealized_pnl),
                "realized_pnl_today": float(realized_pnl_today),
                "positions_detail": (
                    self._serialize_safely(positions_detail)
                    if positions_detail
                    else None
                ),
            }

            result = (
                self.supabase.table("portfolio_snapshots")
                .insert(snapshot_data)
                .execute()
            )

            if result.data:
                logger.debug(f"[DB] Portfolio snapshot recorded")
                return True
            return False

        except Exception as e:
            logger.error(f"[DB] Error inserting portfolio snapshot: {e}")
            return False

    # ========================================================================
    # SYSTEM EVENTS
    # ========================================================================

    def log_system_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        component: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Log system events (startup, shutdown, errors, etc.)"""
        try:
            event_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "component": component,
                "metadata": self._serialize_safely(metadata) if metadata else None,
            }

            result = self.supabase.table("system_events").insert(event_data).execute()
            return bool(result.data)

        except Exception as e:
            logger.error(f"[DB] Error logging system event: {e}")
            return False

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def get_open_trades(self, asset: Optional[str] = None) -> List[Dict]:
        """Get all open trades"""
        try:
            query = self.supabase.table("trades").select("*").eq("status", "open")

            if asset:
                query = query.eq("asset", asset)

            result = query.execute()
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"[DB] Error querying open trades: {e}")
            return []

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict]:
        """Get trade by database ID"""
        try:
            result = (
                self.supabase.table("trades").select("*").eq("id", trade_id).execute()
            )
            return result.data[0] if result.data and len(result.data) > 0 else None

        except Exception as e:
            logger.error(f"[DB] Error querying trade by ID: {e}")
            return None

    def get_trade_by_position_id(self, position_id: str) -> Optional[Dict]:
        """Get trade by position ID"""
        try:
            result = (
                self.supabase.table("trades")
                .select("*")
                .eq("position_id", position_id)
                .execute()
            )
            return result.data[0] if result.data and len(result.data) > 0 else None

        except Exception as e:
            logger.error(f"[DB] Error querying trade by position_id: {e}")
            return None

    def get_trade_by_mt5_ticket(self, mt5_ticket: int) -> Optional[Dict]:
        """Get trade by MT5 ticket number"""
        try:
            result = (
                self.supabase.table("trades")
                .select("*")
                .eq("mt5_ticket", mt5_ticket)
                .execute()
            )
            return result.data[0] if result.data and len(result.data) > 0 else None

        except Exception as e:
            logger.error(f"[DB] Error querying trade by MT5 ticket: {e}")
            return None

    def get_trade_by_binance_order_id(self, binance_order_id: int) -> Optional[Dict]:
        """Get trade by Binance order ID"""
        try:
            result = (
                self.supabase.table("trades")
                .select("*")
                .eq("binance_order_id", binance_order_id)
                .execute()
            )
            return result.data[0] if result.data and len(result.data) > 0 else None

        except Exception as e:
            logger.error(f"[DB] Error querying trade by Binance order ID: {e}")
            return None

    def _get_last_signal_from_db(self, asset: str) -> Optional[Dict]:
        """
        Fetch the most recent signal for an asset from database
        Used for cache warming on startup
        """
        try:
            result = (
                self.supabase.table("signals")
                .select("*")
                .eq("asset", asset)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"[DB] Error fetching last signal: {e}")
            return None

    def _has_signal_changed(self, asset: str, new_signal_data: Dict) -> bool:
        """
        Check if signal has meaningfully changed from last recorded signal

        Args:
            asset: Asset name (BTC/GOLD)
            new_signal_data: New signal data to compare

        Returns:
            True if signal has changed and should be recorded
        """
        # Get cached last signal, or fetch from DB if not in cache
        if asset not in self._last_signals:
            last_signal = self._get_last_signal_from_db(asset)
            if last_signal:
                self._last_signals[asset] = last_signal
            else:
                # No previous signal, this is the first one
                return True
        else:
            last_signal = self._last_signals[asset]

        # Compare key signal components
        key_fields = [
            "signal",  # Most important: -1, 0, 1
            "regime",  # Regime change is significant
            "mr_signal",  # Mean reversion signal
            "tf_signal",  # Trend following signal
        ]

        for field in key_fields:
            if new_signal_data.get(field) != last_signal.get(field):
                logger.info(
                    f"[DB] Signal changed for {asset}: {field} changed from {last_signal.get(field)} to {new_signal_data.get(field)}"
                )
                return True

        # Check if confidence scores changed significantly (>10% change)
        confidence_threshold = 0.1
        confidence_fields = [
            "signal_quality",
            "regime_confidence",
            "mr_confidence",
            "tf_confidence",
        ]

        for field in confidence_fields:
            old_val = last_signal.get(field, 0)
            new_val = new_signal_data.get(field, 0)
            if old_val is not None and new_val is not None:
                if abs(new_val - old_val) > confidence_threshold:
                    logger.info(
                        f"[DB] Signal confidence changed significantly for {asset}: {field} changed by {abs(new_val - old_val):.2%}"
                    )
                    return True

        logger.debug(f"[DB] Signal unchanged for {asset}, skipping insert")
        return False

    def _serialize_safely(self, obj: Any) -> str:
        """
        Safely serialize objects to JSON, handling problematic types
        """

        def convert(item):
            if isinstance(item, (datetime,)):
                return item.isoformat()
            elif isinstance(item, (Decimal,)):
                return float(item)
            elif isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [convert(i) for i in item]
            elif isinstance(item, (int, float, str, bool, type(None))):
                return item
            else:
                return str(item)

        try:
            converted = convert(obj)
            return json.dumps(converted)
        except Exception as e:
            logger.warning(f"[DB] JSON serialization fallback: {e}")
            return json.dumps(str(obj))

    def get_recent_signals(
        self, asset: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Get recent signals"""
        try:
            query = (
                self.supabase.table("signals")
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
            )

            if asset:
                query = query.eq("asset", asset)

            result = query.execute()
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"[DB] Error querying recent signals: {e}")
            return []

    def get_last_signal(self, asset: str) -> Optional[Dict]:
        """
        Get the last recorded signal for an asset (from cache or DB)
        Useful for strategy logic
        """
        if asset in self._last_signals:
            return self._last_signals[asset]
        return self._get_last_signal_from_db(asset)

    def update_signal_execution(
        self, signal_id: int, executed: bool, trade_id: Optional[int] = None
    ) -> bool:
        """Mark signal as executed and link to trade"""
        try:
            update_data = {
                "executed": executed,
                "trade_id": trade_id,
                "execution_time": datetime.now(timezone.utc).isoformat(),
            }

            result = (
                self.supabase.table("signals")
                .update(update_data)
                .eq("id", signal_id)
                .execute()
            )

            # Update cache if present
            for asset, cached_signal in self._last_signals.items():
                if cached_signal.get("id") == signal_id:
                    cached_signal.update(update_data)
                    break

            return bool(result.data)

        except Exception as e:
            logger.error(f"[DB] Error updating signal execution: {e}")
            return False

    def get_performance_stats(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict:
        """Get performance statistics for a date range"""
        try:
            query = self.supabase.table("trades").select("*").eq("status", "closed")

            if start_date:
                query = query.gte("entry_time", start_date.isoformat())
            if end_date:
                query = query.lte("exit_time", end_date.isoformat())

            result = query.execute()

            if not result.data:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                }

            trades = result.data
            total_trades = len(trades)
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] < 0]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(t["pnl"] for t in trades)
            avg_win = (
                sum(t["pnl"] for t in winning_trades) / win_count
                if win_count > 0
                else 0
            )
            avg_loss = (
                sum(t["pnl"] for t in losing_trades) / loss_count
                if loss_count > 0
                else 0
            )

            profit_factor = (
                abs(avg_win * win_count / (avg_loss * loss_count))
                if loss_count > 0 and avg_loss != 0
                else 0
            )

            return {
                "total_trades": total_trades,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
            }

        except Exception as e:
            logger.error(f"[DB] Error calculating performance stats: {e}")
            return {}

    def get_vtm_events_for_trade(self, trade_id: int) -> List[Dict]:
        """Get all VTM events for a specific trade"""
        try:
            result = (
                self.supabase.table("vtm_events")
                .select("*")
                .eq("trade_id", trade_id)
                .order("timestamp")
                .execute()
            )
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"[DB] Error querying VTM events: {e}")
            return []


# ============================================================================
# HELPER FUNCTION FOR INTEGRATION
# ============================================================================


def calculate_daily_summary_from_trades(
    db: TradingDatabaseManager, date: datetime
) -> bool:
    """
    Calculate and insert daily summary from closed trades
    Call this at end of each trading day
    """
    try:
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Get all trades closed today
        trades = (
            db.supabase.table("trades")
            .select("*")
            .eq("status", "closed")
            .gte("exit_time", start_of_day.isoformat())
            .lte("exit_time", end_of_day.isoformat())
            .execute()
        )

        if not trades.data:
            logger.info(f"[DB] No trades closed on {date.date()}")
            return False

        trades_list = trades.data
        total_trades = len(trades_list)
        winning = [t for t in trades_list if t["pnl"] > 0]
        losing = [t for t in trades_list if t["pnl"] < 0]

        btc_trades = len([t for t in trades_list if t["asset"] == "BTC"])
        gold_trades = len([t for t in trades_list if t["asset"] == "GOLD"])

        win_count = len(winning)
        loss_count = len(losing)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(t["pnl"] for t in trades_list)
        avg_win = sum(t["pnl"] for t in winning) / win_count if win_count > 0 else 0
        avg_loss = sum(t["pnl"] for t in losing) / loss_count if loss_count > 0 else 0

        profit_factor = (
            abs(avg_win * win_count / (avg_loss * loss_count))
            if loss_count > 0 and avg_loss != 0
            else 0
        )

        # Get portfolio snapshots for capital tracking
        snapshots = (
            db.supabase.table("portfolio_snapshots")
            .select("*")
            .gte("timestamp", start_of_day.isoformat())
            .lte("timestamp", end_of_day.isoformat())
            .order("timestamp")
            .execute()
        )

        starting_capital = snapshots.data[0]["total_value"] if snapshots.data else 0
        ending_capital = (
            snapshots.data[-1]["total_value"] if snapshots.data else starting_capital
        )

        total_pnl_pct = (
            ((ending_capital - starting_capital) / starting_capital * 100)
            if starting_capital > 0
            else 0
        )

        # Calculate max drawdown for the day
        max_drawdown = 0.0
        if snapshots.data:
            peak = starting_capital
            for snap in snapshots.data:
                value = snap["total_value"]
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return db.insert_daily_summary(
            date=date,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            starting_capital=starting_capital,
            ending_capital=ending_capital,
            btc_trades=btc_trades,
            gold_trades=gold_trades,
        )

    except Exception as e:
        logger.error(f"[DB] Error calculating daily summary: {e}")
        return False
