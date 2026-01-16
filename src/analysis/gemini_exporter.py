# src/analysis/gemini_exporter.py

import json
from src.analysis.storyteller import TradeStoryteller


class GeminiExporter:
    def __init__(self, db_manager):
        self.storyteller = TradeStoryteller(db_manager)

    def generate_report(self, mode: str, date_ref: str) -> str:
        """
        Generates a Markdown report optimized for LLM analysis
        """
        data = self.storyteller.generate_report(mode, date_ref)

        # Dynamic Header based on Timeframe
        title = (
            f"{mode.upper()} TRADE DEBRIEF ({data['start_date']} to {data['end_date']})"
        )

        instruction = ""
        if mode == "daily":
            instruction = "Analyze execution logic, VTM handling, and immediate reaction to news/price. Check for 'Hard Trend Filter' compliance."
        elif mode == "weekly":
            instruction = "Analyze consistency across the week. Did the bot adapt to regime changes? Identify any days with excessive losses."
        elif mode == "monthly":
            instruction = "Analyze long-term profitability and risk management. Are there recurring patterns of failure in specific market conditions? Review Win Rate vs Profit Factor."

        report = f"""
# 🤖 TBOT {title}
# INSTRUCTIONS: {instruction}

## 1. PERIOD OVERVIEW
- **Total Trades:** {data['total_trades']}
- **Period:** {data['start_date']} -- {data['end_date']}

## 2. TRADE EXECUTION LOG
"""
        if not data["trades"]:
            report += "\n*No trades recorded for this period.*\n"
            return report

        for trade in data["trades"]:
            # Formatting PnL with emoji
            pnl_emoji = "🟢" if (trade["pnl"] or 0) > 0 else "🔴"
            status = "OPEN" if not trade["exit_time"] else "CLOSED"

            report += f"""
### {pnl_emoji} Trade #{trade['id']} - {trade['asset']} {trade['side'].upper()}
- **Time:** {trade['entry_time']}
- **Status:** {status}
- **Result:** PnL ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)
- **Strategy Logic:** {trade['narrative']['trigger']}
- **Market Context:** {trade['narrative']['market_context']}
- **AI Verdict:** {trade['narrative']['ai_verdict']}
- **AI Analysis:** {trade['narrative']['ai_analysis']}
- **Management Events:**
"""
            if trade["timeline"]:
                for event in trade["timeline"]:
                    report += (
                        f"  - [{event['time']}] {event['event']} {event['detail']}\n"
                    )
            else:
                report += "  - (No management events recorded)\n"

            report += "\n------------------------------------------------\n"

        report += """
## 3. ANOMALIES & REQUESTS FOR ANALYSIS
- [ ] Check for counter-trend entries in high confidence zones (Should be BLOCKED by Hard Filter).
- [ ] Check for 'Ranging Mode' violations (Max 1 trade enforcement).
- [ ] Evaluate if AI Rejections saved capital or missed opportunities.
"""
        return report
