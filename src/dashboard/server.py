"""
Flask Dashboard Server for Trading Bot
======================================
Serves the real-time dashboard and provides API endpoints
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime, timedelta
from supabase import create_client, Client
import logging

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
SUPABASE_URL = os.getenv('SUPABASE_URL', 'YOUR_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'YOUR_SUPABASE_KEY')

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main dashboard"""
    try:
        # Read the HTML dashboard file
        with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
            dashboard_html = f.read()

        
        # Inject Supabase credentials (client-safe anon key)
        dashboard_html = dashboard_html.replace(
            "'YOUR_SUPABASE_URL'", 
            f"'{SUPABASE_URL}'"
        )
        dashboard_html = dashboard_html.replace(
            "'YOUR_SUPABASE_ANON_KEY'",
            f"'{os.getenv('SUPABASE_ANON_KEY', SUPABASE_KEY)}'"
        )
        
        return render_template_string(dashboard_html)
        
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500


@app.route('/api/stats')
def get_stats():
    """Get current portfolio statistics"""
    try:
        # Get latest portfolio snapshot
        snapshot = supabase.table('portfolio_snapshots')\
            .select('*')\
            .order('timestamp', desc=True)\
            .limit(1)\
            .execute()
        
        # Get performance stats
        trades = supabase.table('trades')\
            .select('*')\
            .eq('status', 'closed')\
            .execute()
        
        stats = {
            'portfolio': snapshot.data[0] if snapshot.data else None,
            'trade_count': len(trades.data) if trades.data else 0,
            'win_rate': calculate_win_rate(trades.data) if trades.data else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/open')
def get_open_trades():
    """Get all open trades"""
    try:
        result = supabase.table('trades')\
            .select('*')\
            .eq('status', 'open')\
            .order('entry_time', desc=True)\
            .execute()
        
        return jsonify({'trades': result.data})
        
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/closed')
def get_closed_trades():
    """Get recent closed trades"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        result = supabase.table('trades')\
            .select('*')\
            .eq('status', 'closed')\
            .order('exit_time', desc=True)\
            .limit(limit)\
            .execute()
        
        return jsonify({'trades': result.data})
        
    except Exception as e:
        logger.error(f"Error getting closed trades: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals')
def get_signals():
    """Get recent trading signals"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        result = supabase.table('signals')\
            .select('*')\
            .order('timestamp', desc=True)\
            .limit(limit)\
            .execute()
        
        return jsonify({'signals': result.data})
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/history')
def get_portfolio_history():
    """Get portfolio value history"""
    try:
        hours = request.args.get('hours', 24, type=int)
        start_time = datetime.now() - timedelta(hours=hours)
        
        result = supabase.table('portfolio_snapshots')\
            .select('timestamp, total_value, unrealized_pnl')\
            .gte('timestamp', start_time.isoformat())\
            .order('timestamp', desc=False)\
            .execute()
        
        return jsonify({'history': result.data})
        
    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    try:
        days = request.args.get('days', 7, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        # Get trades in date range
        trades = supabase.table('trades')\
            .select('*')\
            .eq('status', 'closed')\
            .gte('exit_time', start_date.isoformat())\
            .execute()
        
        if not trades.data:
            return jsonify({
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            })
        
        # Calculate metrics
        total_trades = len(trades.data)
        winning_trades = [t for t in trades.data if t['pnl'] > 0]
        losing_trades = [t for t in trades.data if t['pnl'] < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        total_pnl = sum(t['pnl'] for t in trades.data)
        avg_win = sum(t['pnl'] for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / loss_count if loss_count > 0 else 0
        
        profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if loss_count > 0 and avg_loss != 0 else 0
        
        return jsonify({
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': (win_count / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': max(trades.data, key=lambda x: x['pnl'])['pnl'] if trades.data else 0,
            'worst_trade': min(trades.data, key=lambda x: x['pnl'])['pnl'] if trades.data else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/events')
def get_system_events():
    """Get recent system events"""
    try:
        limit = request.args.get('limit', 50, type=int)
        severity = request.args.get('severity')
        
        query = supabase.table('system_events')\
            .select('*')\
            .order('timestamp', desc=True)\
            .limit(limit)
        
        if severity:
            query = query.eq('severity', severity)
        
        result = query.execute()
        
        return jsonify({'events': result.data})
        
    except Exception as e:
        logger.error(f"Error getting system events: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        result = supabase.table('trades').select('id').limit(1).execute()
        
        # Get last activity
        snapshot = supabase.table('portfolio_snapshots')\
            .select('timestamp')\
            .order('timestamp', desc=True)\
            .limit(1)\
            .execute()
        
        last_activity = snapshot.data[0]['timestamp'] if snapshot.data else None
        
        # Determine if bot is active (activity in last 10 minutes)
        is_active = False
        if last_activity:
            last_time = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
            is_active = (datetime.now(last_time.tzinfo) - last_time).total_seconds() < 600
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'bot_active': is_active,
            'last_activity': last_activity,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_win_rate(trades):
    """Calculate win rate from trades"""
    if not trades:
        return 0
    winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
    return (winning / len(trades) * 100)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Check configuration
    if SUPABASE_URL == 'YOUR_SUPABASE_URL':
        logger.error("❌ Please set SUPABASE_URL in environment or code")
        exit(1)
    
    logger.info("=" * 80)
    logger.info("🚀 TOM's Trading Bot Dashboard Server")
    logger.info("=" * 80)
    logger.info(f"Dashboard: http://localhost:5000")
    logger.info(f"API Docs:  http://localhost:5000/api/health")
    logger.info("=" * 80)
    
    # Run server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True,  # Set to False in production
        threaded=True
    )