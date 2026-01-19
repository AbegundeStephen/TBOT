"""
Multi-Timeframe Regime Detector - Usage Examples
==================================================
Shows how to use the MTF detector in various scenarios
"""

from src.execution.mtf_regime_detector import MultiTimeFrameRegimeDetector
from src.execution.mtf_integration import MTFRegimeIntegration
from src.data.data_manager import DataManager
import json


# ============================================================================
# Example 1: Basic Usage - Analyze Regime for BTC
# ============================================================================
def example_basic_analysis():
    """Basic regime analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic MTF Regime Analysis")
    print("="*70)
    
    # Load config
    with open("config/config.json") as f:
        config = json.load(f)
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Create detector for BTC
    detector = MultiTimeFrameRegimeDetector(
        data_manager=data_manager,
        asset_type="BTC"
    )
    
    # Analyze regime
    regime = detector.analyze_regime(
        symbol="BTCUSDT",
        exchange="binance"
    )
    
    # Print summary
    print(detector.get_regime_summary(regime))
    
    # Access specific fields
    print(f"\nQuick Access:")
    print(f"  Is Bull Market? {regime.consensus_regime.value in ['bull', 'strong_bull']}")
    print(f"  Can Trade Counter-Trend? {regime.allow_counter_trend}")
    print(f"  Recommended Mode: {regime.recommended_mode}")
    print(f"  Max Positions: {regime.suggested_max_positions}")


# ============================================================================
# Example 2: Use in Trading Decision
# ============================================================================
def example_trading_decision():
    """Use MTF regime in trading logic"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Using MTF in Trading Decision")
    print("="*70)
    
    # Assume you have these from your trading bot
    signal = 1  # BUY signal from aggregator
    asset_name = "BTC"
    
    # Get MTF regime (from cache)
    regime_data = bot._current_regime_data.get(asset_name)
    
    if regime_data:
        # Check 1: Counter-trend filter
        if not regime_data['allow_counter_trend']:
            is_counter = (signal == 1 and not regime_data['is_bull']) or \
                        (signal == -1 and regime_data['is_bull'])
            
            if is_counter:
                print(f"❌ Trade BLOCKED: Counter-trend not allowed")
                print(f"   Signal: {'BUY' if signal == 1 else 'SELL'}")
                print(f"   Regime: {'BULL' if regime_data['is_bull'] else 'BEAR'}")
                return False
        
        # Check 2: Position limits
        max_pos = regime_data['max_positions']
        current_pos = 1  # Example
        
        if current_pos >= max_pos:
            print(f"❌ Trade BLOCKED: Max positions ({current_pos}/{max_pos})")
            return False
        
        # Check 3: Risk adjustments
        risk_mult = 1.0
        if regime_data['risk_level'] == 'high':
            risk_mult = 0.7
            print(f"⚠️  Position size reduced to 70% (high risk)")
        elif regime_data['risk_level'] == 'low':
            risk_mult = 1.2
            print(f"✅ Position size increased to 120% (low risk)")
        
        print(f"\n✓ Trade APPROVED:")
        print(f"  Regime: {regime_data['regime'].upper()}")
        print(f"  Confidence: {regime_data['confidence']:.2%}")
        print(f"  Risk Multiplier: {risk_mult:.1%}")
        
        return True


# ============================================================================
# Example 3: Compare Multiple Assets
# ============================================================================
def example_multi_asset_analysis():
    """Analyze regime for multiple assets"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Multi-Asset Regime Analysis")
    print("="*70)
    
    with open("config/config.json") as f:
        config = json.load(f)
    
    data_manager = DataManager(config)
    
    assets = [
        ("BTC", "BTCUSDT", "binance"),
        ("GOLD", "XAUUSD", "mt5")
    ]
    
    results = {}
    
    for asset_name, symbol, exchange in assets:
        detector = MultiTimeFrameRegimeDetector(
            data_manager=data_manager,
            asset_type=asset_name
        )
        
        regime = detector.analyze_regime(
            symbol=symbol,
            exchange=exchange
        )
        
        results[asset_name] = regime
    
    # Compare regimes
    print("\n" + "="*70)
    print("REGIME COMPARISON")
    print("="*70)
    
    for asset_name, regime in results.items():
        print(f"\n{asset_name}:")
        print(f"  Consensus:  {regime.consensus_regime.value.upper()}")
        print(f"  Confidence: {regime.consensus_confidence:.2%}")
        print(f"  Agreement:  {regime.timeframe_agreement:.2%}")
        print(f"  Mode:       {regime.recommended_mode.upper()}")
        print(f"  Risk:       {regime.risk_level.upper()}")


# ============================================================================
# Example 4: Cache Management
# ============================================================================
def example_cache_management():
    """Demonstrate cache usage"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Cache Management")
    print("="*70)
    
    with open("config/config.json") as f:
        config = json.load(f)
    
    data_manager = DataManager(config)
    detector = MultiTimeFrameRegimeDetector(
        data_manager=data_manager,
        asset_type="BTC"
    )
    
    # First call - fetches fresh data
    print("\n1st call (fresh data):")
    regime1 = detector.analyze_regime("BTCUSDT", "binance")
    print(f"   Regime: {regime1.consensus_regime.value}")
    
    # Second call - uses cache
    print("\n2nd call (from cache):")
    regime2 = detector.analyze_regime("BTCUSDT", "binance")
    print(f"   Regime: {regime2.consensus_regime.value}")
    print(f"   Same object? {regime1 is regime2}")
    
    # Force refresh
    print("\n3rd call (force refresh):")
    regime3 = detector.analyze_regime("BTCUSDT", "binance", force_refresh=True)
    print(f"   Regime: {regime3.consensus_regime.value}")
    print(f"   Same object? {regime1 is regime3}")
    
    # Check cache directly
    print("\n4th call (check cache):")
    cached = detector.get_cached_regime("BTCUSDT", "binance")
    if cached:
        print(f"   Cache hit! Age: {(datetime.now() - cached.timestamp).total_seconds():.0f}s")
    else:
        print(f"   Cache miss")
    
    # Clear cache
    print("\n5th call (after cache clear):")
    detector.clear_cache()
    cached = detector.get_cached_regime("BTCUSDT", "binance")
    print(f"   Cache after clear: {'HIT' if cached else 'MISS'}")


# ============================================================================
# Example 5: Database Integration
# ============================================================================
def example_database_integration():
    """Show database integration"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Database Integration")
    print("="*70)
    
    # Assume you have these initialized
    # mtf_integration = MTFRegimeIntegration(
    #     data_manager=data_manager,
    #     db_manager=db_manager,
    #     ai_validator=ai_validator,
    #     telegram_bot=telegram_bot
    # )
    
    # Analyze and auto-log to database
    regime_data = mtf_integration.get_regime_for_trading(
        asset_name="BTC",
        symbol="BTCUSDT",
        exchange="binance"
    )
    
    print(f"\n✓ Regime analyzed and logged to database:")
    print(f"  Regime: {regime_data['regime']}")
    print(f"  Confidence: {regime_data['confidence']:.2%}")
    print(f"  Recommended Mode: {regime_data['recommended_mode']}")
    
    # The integration automatically:
    # 1. Logs to Supabase (mtf_regime_analysis table)
    # 2. Updates AI validator context
    # 3. Sends Telegram notification (if significant change)


# ============================================================================
# Example 6: AI Validator Context
# ============================================================================
def example_ai_validator_integration():
    """Show how MTF updates AI validator"""
    print("\n" + "="*70)
    print("EXAMPLE 6: AI Validator Integration")
    print("="*70)
    
    # After MTF analysis runs, AI validator has context
    if hasattr(ai_validator, 'mtf_regime_context'):
        btc_context = ai_validator.mtf_regime_context.get('BTC')
        
        if btc_context:
            print(f"\nAI Validator now has MTF context:")
            print(f"  Regime: {btc_context['regime']}")
            print(f"  Confidence: {btc_context['confidence']:.2%}")
            print(f"  Allow Counter-Trend: {btc_context['allow_counter_trend']}")
            print(f"  Risk Level: {btc_context['risk_level']}")
            
            print(f"\nAI can now:")
            print(f"  - Adjust S/R thresholds based on volatility")
            print(f"  - Require higher pattern confidence in uncertain regimes")
            print(f"  - Block counter-trend patterns when regime is strong")


# ============================================================================
# Example 7: Convert to Dictionary
# ============================================================================
def example_to_dict():
    """Show how to convert regime to dict"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Regime to Dictionary")
    print("="*70)
    
    with open("config/config.json") as f:
        config = json.load(f)
    
    data_manager = DataManager(config)
    detector = MultiTimeFrameRegimeDetector(
        data_manager=data_manager,
        asset_type="BTC"
    )
    
    regime = detector.analyze_regime("BTCUSDT", "binance")
    
    # Convert to dict (useful for JSON serialization)
    regime_dict = regime.to_dict()
    
    print(f"\nRegime as dictionary (showing first 10 keys):")
    for i, (key, value) in enumerate(list(regime_dict.items())[:10]):
        print(f"  {key}: {value}")
    
    print(f"\n... and {len(regime_dict) - 10} more fields")
    
    # Can be saved to JSON
    import json
    json_str = json.dumps(regime_dict, indent=2, default=str)
    print(f"\nJSON serializable? {len(json_str) > 0}")


# ============================================================================
# Example 8: Scheduled Analysis
# ============================================================================
def example_scheduled_analysis():
    """Show how to schedule MTF analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Scheduled Analysis (Pseudocode)")
    print("="*70)
    
    print("""
    In your TradingBot.start() method:
    
    # Schedule MTF analysis every 4 hours
    schedule.every(4).hours.do(self.run_mtf_regime_analysis)
    
    # Or manually set time
    schedule.every().day.at("00:00").do(self.run_mtf_regime_analysis)
    schedule.every().day.at("04:00").do(self.run_mtf_regime_analysis)
    schedule.every().day.at("08:00").do(self.run_mtf_regime_analysis)
    schedule.every().day.at("12:00").do(self.run_mtf_regime_analysis)
    schedule.every().day.at("16:00").do(self.run_mtf_regime_analysis)
    schedule.every().day.at("20:00").do(self.run_mtf_regime_analysis)
    
    This ensures:
    - Regime is always current
    - Database has full history
    - AI validator has latest context
    - Trading decisions use fresh regime data
    """)


# ============================================================================
# Run Examples
# ============================================================================
if __name__ == "__main__":
    # Uncomment to run specific examples
    
    # example_basic_analysis()
    # example_trading_decision()
    # example_multi_asset_analysis()
    # example_cache_management()
    # example_to_dict()
    # example_scheduled_analysis()
    
    print("\n" + "="*70)
    print("Examples complete! Uncomment functions to run them.")
    print("="*70)