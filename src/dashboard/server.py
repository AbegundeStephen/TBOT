"""
Flask Dashboard Server for Trading Bot
======================================
Serves the real-time dashboard and provides API endpoints
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client, Client
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))  # src/dashboard
src_dir = os.path.dirname(current_dir)  # src
project_root = os.path.dirname(src_dir)  # TBOT root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.storyteller import TradeStoryteller
from src.analysis.gemini_exporter import GeminiExporter
from src.database.database_manager import TradingDatabaseManager

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for real-time updates


# Load from environment or config
SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Initialize DB Manager (For Analysis Tools)
db_manager = TradingDatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# ROUTES
# ============================================================================


@app.route("/")
def index():
    """Serve the main dashboard"""
    try:
        # Read the HTML dashboard file
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            dashboard_html = f.read()

        # Inject Supabase credentials (client-safe anon key)
        dashboard_html = dashboard_html.replace(
            "'YOUR_SUPABASE_URL'", f"'{SUPABASE_URL}'"
        )
        dashboard_html = dashboard_html.replace(
            "'YOUR_SUPABASE_ANON_KEY'",
            f"'{os.getenv('SUPABASE_ANON_KEY', SUPABASE_KEY)}'",
        )

        return render_template_string(dashboard_html)

    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return jsonify({"error": "Failed to load dashboard"}), 500


@app.route("/debrief")
def debrief_page():
    """Serve the Daily/Weekly Debrief Page"""
    try:
        # Try multiple possible locations for the template
        possible_paths = [
            "src/dashboard/templates/daily_recap.html",
            "templates/daily_recap.html",
            os.path.join(current_dir, "templates", "daily_recap.html"),
            os.path.join(
                project_root, "src", "dashboard", "templates", "daily_recap.html"
            ),
        ]

        template_path = None
        for path in possible_paths:
            if os.path.exists(path):
                template_path = path
                logger.info(f"Found template at: {path}")
                break

        if not template_path:
            error_msg = f"daily_recap.html not found. Searched in:\n"
            error_msg += "\n".join(f"  - {p}" for p in possible_paths)
            logger.error(error_msg)
            return error_msg, 404

        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        return render_template_string(content)

    except Exception as e:
        logger.error(f"Error serving debrief: {e}")
        return f"Error loading debrief page: {str(e)}", 500


# ============================================================================
# API - ANALYSIS & FEEDBACK LOOP
# ============================================================================


@app.route("/api/debrief_data")
def get_debrief_data():
    """Get structured data for the visual debrief"""
    try:
        date_ref = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        mode = request.args.get("mode", "daily")

        storyteller = TradeStoryteller(db_manager)
        data = storyteller.generate_report(mode, date_ref)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Debrief data error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_gemini")
def export_gemini_report():
    """Generate text prompt for Gemini analysis"""
    try:
        date_ref = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        mode = request.args.get("mode", "daily")

        exporter = GeminiExporter(db_manager)
        report_text = exporter.generate_report(mode, date_ref)

        return jsonify({"status": "success", "report": report_text})
    except Exception as e:
        logger.error(f"Gemini export error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/<asset>")
def get_history(asset):
    """
    Serves candle data for the chart from local CSVs.
    Used by the Lightweight Charts in /debrief
    """
    start = request.args.get("start")
    end = request.args.get("end")

    # Map friendly names to CSV filenames
    filename_map = {
        "BTC": "BTCUSDT_1h.csv",
        "GOLD": "XAUUSDm_1h.csv",
        "XAU": "XAUUSDm_1h.csv",
    }

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_file = filename_map.get(asset.upper())
    if not csv_file:
        return jsonify([])

    # Check multiple locations
    paths = [
        os.path.join(BASE_DIR, "data", "raw", csv_file),
    ]

    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] Checking paths:")
    csv_path = None
    for p in paths:
        print(" -", p)
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        print(f"[ERROR] CSV file not found for {asset}: {csv_file}")
        return jsonify([])

    try:
        df = pd.read_csv(csv_path)

        # Handle different column names for timestamp
        timestamp_col = None
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                timestamp_col = col
                break

        if not timestamp_col:
            print(f"[ERROR] No timestamp column in {csv_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return jsonify([])

        # Rename to standard 'timestamp' for consistency
        if timestamp_col != "timestamp":
            df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

        # Convert to datetime with UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Parse start/end parameters
        if start:
            try:
                start_dt = pd.to_datetime(start, utc=True)
                df = df[df["timestamp"] >= start_dt]
            except Exception as e:
                print(f"[WARNING] Could not parse start date '{start}': {e}")

        if end:
            try:
                end_dt = pd.to_datetime(end, utc=True)
                df = df[df["timestamp"] <= end_dt]
            except Exception as e:
                print(f"[WARNING] Could not parse end date '{end}': {e}")

        print(f"[INFO] Filtered {len(df)} candles for {asset} from {start} to {end}")

        if len(df) == 0:
            print(
                f"[WARNING] No data in date range. CSV range: {pd.to_datetime(pd.read_csv(csv_path)[timestamp_col]).min()} to {pd.to_datetime(pd.read_csv(csv_path)[timestamp_col]).max()}"
            )

        # Format for Lightweight Charts (Unix timestamp in SECONDS)
        chart_data = []
        for _, row in df.iterrows():
            chart_data.append(
                {
                    "time": int(row["timestamp"].timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )

        return jsonify(chart_data)

    except Exception as e:
        print(f"[ERROR] Failed to load history for {asset}: {e}")
        import traceback

        traceback.print_exc()
        return jsonify([])


@app.route("/api/stats")
def get_stats():
    """Get current portfolio statistics"""
    try:
        # Get latest portfolio snapshot
        snapshot = (
            supabase.table("portfolio_snapshots")
            .select("*")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        # Get performance stats
        trades = supabase.table("trades").select("*").eq("status", "closed").execute()

        stats = {
            "portfolio": snapshot.data[0] if snapshot.data else None,
            "trade_count": len(trades.data) if trades.data else 0,
            "win_rate": calculate_win_rate(trades.data) if trades.data else 0,
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades/open")
def get_open_trades():
    """Get all open trades"""
    try:
        result = (
            supabase.table("trades")
            .select("*")
            .eq("status", "open")
            .order("entry_time", desc=True)
            .execute()
        )

        return jsonify({"trades": result.data})

    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades/closed")
def get_closed_trades():
    """Get recent closed trades"""
    try:
        limit = request.args.get("limit", 20, type=int)

        result = (
            supabase.table("trades")
            .select("*")
            .eq("status", "closed")
            .order("exit_time", desc=True)
            .limit(limit)
            .execute()
        )

        return jsonify({"trades": result.data})

    except Exception as e:
        logger.error(f"Error getting closed trades: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regime/<asset>")
def get_regime(asset):
    """
    Fetches the latest multi-timeframe regime analysis for a given asset and date.
    If no data exists for the selected date, returns the most recent data.
    """
    try:
        date = request.args.get("date")

        # Query regime data for the selected asset
        query = (
            supabase.table("mtf_regime_analysis").select("*").eq("asset", asset.upper())
        )

        if date:
            # Convert the date to a range for the day (start and end of the day)
            start_of_day = f"{date}T00:00:00+00:00"
            end_of_day = f"{date}T23:59:59+00:00"

            # Filter by timestamp range
            query = query.gte("timestamp", start_of_day).lte("timestamp", end_of_day)

        # Order by timestamp (descending) and fetch the latest record
        result = query.order("timestamp", desc=True).limit(1).execute()

        if not result.data:
            # If no data for the selected date, fetch the most recent data
            result = (
                supabase.table("mtf_regime_analysis")
                .select("*")
                .eq("asset", asset.upper())
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                return jsonify({"error": f"No regime data found for {asset}"}), 404

        regime_data = result.data[0]

        # Format the response
        response = {
            "asset": regime_data["asset"],
            "timestamp": regime_data["timestamp"],
            "consensus": {
                "regime": regime_data["consensus_regime"],
                "confidence": regime_data["consensus_confidence"],
                "timeframe_agreement": regime_data["timeframe_agreement"],
                "trend_coherence": regime_data["trend_coherence"],
            },
            "risk": {
                "level": regime_data["risk_level"],
                "volatility": regime_data["volatility_regime"],
            },
            "trading_implications": {
                "recommended_mode": regime_data["recommended_mode"],
                "allow_counter_trend": regime_data["allow_counter_trend"],
                "suggested_max_positions": regime_data["suggested_max_positions"],
            },
            "timeframes": {
                "1h": {
                    "regime": regime_data["h1_regime"],
                    "confidence": regime_data["h1_confidence"],
                    "trend_strength": regime_data["h1_trend_strength"],
                    "trend_direction": regime_data["h1_trend_direction"],
                    "adx": regime_data["h1_adx"],
                    "rsi": regime_data["h1_rsi"],
                    "ema_diff_pct": regime_data["h1_ema_diff_pct"],
                },
                "4h": {
                    "regime": regime_data["h4_regime"],
                    "confidence": regime_data["h4_confidence"],
                    "trend_strength": regime_data["h4_trend_strength"],
                    "trend_direction": regime_data["h4_trend_direction"],
                    "adx": regime_data["h4_adx"],
                    "rsi": regime_data["h4_rsi"],
                    "ema_diff_pct": regime_data["h4_ema_diff_pct"],
                },
                "1d": {
                    "regime": regime_data["d1_regime"],
                    "confidence": regime_data["d1_confidence"],
                    "trend_strength": regime_data["d1_trend_strength"],
                    "trend_direction": regime_data["d1_trend_direction"],
                    "adx": regime_data["d1_adx"],
                    "rsi": regime_data["d1_rsi"],
                    "ema_diff_pct": regime_data["d1_ema_diff_pct"],
                },
            },
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error fetching regime data for {asset}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals")
def get_signals():
    """Get recent trading signals"""
    try:
        limit = request.args.get("limit", 20, type=int)

        result = (
            supabase.table("signals")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return jsonify({"signals": result.data})

    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio/history")
def get_portfolio_history():
    """Get portfolio value history"""
    try:
        hours = request.args.get("hours", 24, type=int)
        start_time = datetime.now() - timedelta(hours=hours)

        result = (
            supabase.table("portfolio_snapshots")
            .select("timestamp, total_value, unrealized_pnl")
            .gte("timestamp", start_time.isoformat())
            .order("timestamp", desc=False)
            .execute()
        )

        return jsonify({"history": result.data})

    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance")
def get_performance():
    """Get performance metrics"""
    try:
        days = request.args.get("days", 7, type=int)
        start_date = datetime.now() - timedelta(days=days)

        # Get trades in date range
        trades = (
            supabase.table("trades")
            .select("*")
            .eq("status", "closed")
            .gte("exit_time", start_date.isoformat())
            .execute()
        )

        if not trades.data:
            return jsonify(
                {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "profit_factor": 0,
                }
            )

        # Calculate metrics
        total_trades = len(trades.data)
        winning_trades = [t for t in trades.data if t["pnl"] > 0]
        losing_trades = [t for t in trades.data if t["pnl"] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)

        total_pnl = sum(t["pnl"] for t in trades.data)
        avg_win = (
            sum(t["pnl"] for t in winning_trades) / win_count if win_count > 0 else 0
        )
        avg_loss = (
            sum(t["pnl"] for t in losing_trades) / loss_count if loss_count > 0 else 0
        )

        profit_factor = (
            abs(avg_win * win_count / (avg_loss * loss_count))
            if loss_count > 0 and avg_loss != 0
            else 0
        )

        return jsonify(
            {
                "total_trades": total_trades,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "win_rate": (win_count / total_trades * 100) if total_trades > 0 else 0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "best_trade": (
                    max(trades.data, key=lambda x: x["pnl"])["pnl"]
                    if trades.data
                    else 0
                ),
                "worst_trade": (
                    min(trades.data, key=lambda x: x["pnl"])["pnl"]
                    if trades.data
                    else 0
                ),
            }
        )

    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/system/events")
def get_system_events():
    """Get recent system events"""
    try:
        limit = request.args.get("limit", 50, type=int)
        severity = request.args.get("severity")

        query = (
            supabase.table("system_events")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
        )

        if severity:
            query = query.eq("severity", severity)

        result = query.execute()

        return jsonify({"events": result.data})

    except Exception as e:
        logger.error(f"Error getting system events: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/templates/<path:filename>")
def serve_template(filename):
    """Serve documentation template files"""
    try:
        file_path = os.path.join("templates", filename)

        # Security check - ensure file is in templates directory
        if not os.path.abspath(file_path).startswith(os.path.abspath("templates")):
            return jsonify({"error": "Invalid file path"}), 403

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content, 200, {"Content-Type": "text/html; charset=utf-8"}

    except Exception as e:
        logger.error(f"Error serving template {filename}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        result = supabase.table("trades").select("id").limit(1).execute()

        # Get last activity
        snapshot = (
            supabase.table("portfolio_snapshots")
            .select("timestamp")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        last_activity = snapshot.data[0]["timestamp"] if snapshot.data else None

        # Determine if bot is active (activity in last 10 minutes)
        is_active = False
        if last_activity:
            last_time = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            is_active = (
                datetime.now(last_time.tzinfo) - last_time
            ).total_seconds() < 600

        return jsonify(
            {
                "status": "healthy",
                "database": "connected",
                "bot_active": is_active,
                "last_activity": last_activity,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_win_rate(trades):
    """Calculate win rate from trades"""
    if not trades:
        return 0
    winning = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return winning / len(trades) * 100


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check configuration
    if SUPABASE_URL == "YOUR_SUPABASE_URL":
        logger.error("❌ Please set SUPABASE_URL in environment or code")
        exit(1)

    logger.info("=" * 80)
    logger.info("🚀 TOM's Trading Bot Dashboard Server")
    logger.info("=" * 80)
    logger.info(f"Dashboard: http://localhost:5000")
    logger.info(f"API Docs:  http://localhost:5000/api/health")
    logger.info(f"Debrief:   http://localhost:5000/debrief")
    logger.info("=" * 80)

    # Run server
    app.run(
        host="0.0.0.0",  # Allow external connections
        port=5000,
        debug=True,  # Set to False in production
        threaded=True,
    )
