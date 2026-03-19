# src/analysis/storyteller.py

import logging
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from src.database.database_manager import TradingDatabaseManager
import calendar

logger = logging.getLogger(__name__)


class TradeStoryteller:
    """
    Reconstructs the full narrative of trades for the Daily/Weekly/Monthly Debrief.
    Combines: Market Structure + Strategy Logic + AI Validation + Execution
    """

    def __init__(self, db_manager: TradingDatabaseManager):
        self.db = db_manager

    def generate_report(self, mode: str, date_ref: str) -> Dict:
        """
        Generates the full animation payload for a specific period.
        mode: 'daily', 'weekly', 'monthly'
        date_ref: 'YYYY-MM-DD'
        """
        start_dt, end_dt = self._calculate_range(mode, date_ref)

        # Query DB for the range
        response = (
            self.db.supabase.table("trades")
            .select("*")
            .gte("entry_time", start_dt.isoformat())
            .lte("entry_time", end_dt.isoformat())
            .order("entry_time")
            .execute()
        )

        trades = response.data if response.data else []
        recap_data = []

        for trade in trades:
            story = self._build_trade_story(trade)
            if story:
                recap_data.append(story)

        return {
            "period_type": mode,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "total_trades": len(recap_data),
            "trades": recap_data,
        }

    def _calculate_range(self, mode, date_str):
        """Calculate start/end datetimes based on mode"""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            dt = datetime.now()

        # Set to beginning of that day
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

        if mode == "weekly":
            # Start = Monday, End = Sunday
            start = dt - timedelta(days=dt.weekday())
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif mode == "monthly":
            # Start = 1st, End = Last Day
            start = dt.replace(day=1)
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            end = dt.replace(day=last_day, hour=23, minute=59, second=59)
        else:  # Daily
            start = dt
            end = dt.replace(hour=23, minute=59, second=59)

        return start, end

    def _build_trade_story(self, trade: Dict) -> Dict:
        """
        Reconstructs the specific logic for ONE trade.
        """
        trade_id = trade["id"]
        asset = trade["asset"]

        # --- A. Fetch The Signal (The "Why") ---
        signal_data = {}
        # Try to find signal linked to trade
        sig_response = (
            self.db.supabase.table("signals")
            .select("*")
            .eq("trade_id", trade_id)
            .execute()
        )
        if sig_response.data:
            signal_data = sig_response.data[0]
        else:
            # Fallback: Find signal near entry time if linkage missing
            entry_time = datetime.fromisoformat(
                trade["entry_time"].replace("Z", "+00:00")
            )
            start_window = (entry_time - timedelta(minutes=15)).isoformat()
            end_window = (entry_time + timedelta(minutes=5)).isoformat()

            sig_response = (
                self.db.supabase.table("signals")
                .select("*")
                .eq("asset", asset)
                .gte("timestamp", start_window)
                .lte("timestamp", end_window)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            if sig_response.data:
                signal_data = sig_response.data[0]

        # --- B. Fetch AI Details (The "Validator") ---
        ai_narrative = "AI Validation was disabled or not recorded."
        ai_verdict = "N/A"

        if signal_data.get("ai_details"):
            try:
                ai_json = (
                    json.loads(signal_data["ai_details"])
                    if isinstance(signal_data["ai_details"], str)
                    else signal_data["ai_details"]
                )
                ai_action = ai_json.get("action", "unknown")

                # Construct AI Narrative
                if ai_action == "approved":
                    ai_verdict = "✅ PASSED"
                    ai_narrative = "AI confirmed the signal. "
                    if ai_json.get("pattern_detected"):
                        # Ensure confidence is a float
                        conf = float(ai_json.get("pattern_confidence") or 0.0)
                        ai_narrative += f"Identified {ai_json.get('pattern_name', 'pattern')} ({conf:.1%} conf)."
                elif ai_action == "bypassed_strong_signal":
                    ai_verdict = "⚠️ BYPASSED (Strong)"
                    ai_narrative = "Signal was so strong that AI validation was skipped to ensure entry."
                elif ai_action == "bypassed":
                    ai_verdict = "⚠️ BYPASSED (Circuit Breaker)"
                    ai_narrative = "AI Circuit Breaker was active (too many rejections previously)."
                elif ai_action == "rejected":
                    ai_verdict = "🛑 BLOCKED"
                    ai_narrative = f"AI tried to block this trade: {', '.join(ai_json.get('rejection_reasons', []))}."
            except Exception as e:
                ai_narrative = f"Could not parse AI details: {str(e)}"

        # --- C. Council/Scalper Logic ---
        strategy_logic = signal_data.get("reasoning", "Standard Strategy Entry")

        # Check metadata for Scalper override info
        meta = {}
        if signal_data.get("metadata"):
            meta = (
                json.loads(signal_data["metadata"])
                if isinstance(signal_data["metadata"], str)
                else signal_data["metadata"]
            )

        if "max_trades_override" in meta:
            strategy_logic += f" [Ranging Mode: Forced Max 1 Trade]"

        # --- D. Market Structure Context ---
        regime = signal_data.get('regime')
        conf = signal_data.get('regime_confidence')
        
        if not regime or not conf:
            # Fallback: Fetch from MTF table if signal linkage is missing
            try:
                entry_time = datetime.fromisoformat(trade["entry_time"].replace("Z", "+00:00"))
                reg_response = (
                    self.db.supabase.table("mtf_regime_analysis")
                    .select("consensus_regime, consensus_confidence")
                    .eq("asset", asset)
                    .lte("timestamp", entry_time.isoformat())
                    .order("timestamp", desc=True)
                    .limit(1)
                    .execute()
                )
                if reg_response.data:
                    regime = reg_response.data[0]["consensus_regime"]
                    conf = reg_response.data[0]["consensus_confidence"]
            except:
                pass

        market_context = f"Regime: {regime or 'Unknown'} (Conf: {float(conf or 0):.2f})"

        # --- E. VTM Events (The "Journey") ---
        vtm_events = self.db.get_vtm_events_for_trade(trade_id)
        formatted_events = []
        for e in vtm_events:
            ts = e["timestamp"].split("T")[1][:5]  # HH:MM
            formatted_events.append(
                {
                    "time": ts,
                    "event": e["event_type"],
                    "detail": (
                        f"{e.get('old_value', '')} -> {e.get('new_value', '')}"
                        if e.get("new_value")
                        else ""
                    ),
                }
            )

        # Calculate timestamps for chart
        entry_dt = datetime.fromisoformat(trade["entry_time"].replace("Z", "+00:00"))
        exit_dt = (
            datetime.fromisoformat(trade["exit_time"].replace("Z", "+00:00"))
            if trade.get("exit_time")
            else datetime.now(timezone.utc)
        )

        chart_start = (entry_dt - timedelta(hours=6)).isoformat()
        chart_end = (exit_dt + timedelta(hours=2)).isoformat()

        # Fix: Ensure PnL is always float (handles None/NULL from DB)
        pnl_val = float(trade.get("pnl") if trade.get("pnl") is not None else 0.0)
        pnl_pct_val = float(
            trade.get("pnl_pct") if trade.get("pnl_pct") is not None else 0.0
        )

        return {
            "id": trade_id,
            "asset": asset,
            "side": trade["side"],
            "entry_time": trade["entry_time"],
            "exit_time": trade.get("exit_time"),
            "pnl": pnl_val,  # ✅ FIXED
            "pnl_pct": pnl_pct_val,  # ✅ FIXED
            "chart_data_start": chart_start,
            "chart_data_end": chart_end,
            "narrative": {
                "trigger": strategy_logic,
                "ai_verdict": ai_verdict,
                "ai_analysis": ai_narrative,
                "market_context": market_context,
            },
            "timeline": formatted_events,
        }
