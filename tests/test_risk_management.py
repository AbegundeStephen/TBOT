# ============================================================================
"""
Add these tests to tests/test_risk_management.py
Run with: pytest tests/test_risk_management.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


def test_risk_based_sizing_btc_long():
    """Test BTC long position sizing with 1.5% risk"""
    
    # Setup
    config = {
        "portfolio": {
            "target_risk_per_trade": 0.015,
            "max_risk_per_trade": 0.020,
        },
        "assets": {
            "BTC": {
                "min_position_usd": 100,
                "max_position_usd": 6000,
                "risk": {
                    "min_stop_distance_pct": 0.015,
                    "max_stop_distance_pct": 0.06,
                }
            }
        }
    }
    
    portfolio_manager = Mock()
    portfolio_manager.current_capital = 10000
    portfolio_manager.positions = {}
    
    sizer = HybridPositionSizer(config, portfolio_manager)
    
    # Test case: BTC at $95,000, stop at $92,000 (3.16% away)
    entry_price = 95000
    stop_loss_price = 92000
    
    position_size, metadata = sizer.calculate_size_risk_based(
        asset="BTC",
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        signal=1,
        confidence_score=0.65,
    )
    
    # Verify calculations
    expected_risk_usd = 10000 * 0.015  # $150
    stop_distance_pct = (95000 - 92000) / 95000  # 0.0316 (3.16%)
    expected_position = expected_risk_usd / stop_distance_pct  # $4,746
    
    assert position_size == pytest.approx(expected_position, rel=0.01)
    assert metadata["actual_risk_usd"] == pytest.approx(150, rel=0.01)
    assert metadata["actual_risk_pct"] == pytest.approx(1.5, rel=0.1)
    
    print(f"✅ Test passed:")
    print(f"  Position: ${position_size:,.2f}")
    print(f"  Risk: ${metadata['actual_risk_usd']:,.2f} ({metadata['actual_risk_pct']:.2%})")


def test_risk_based_sizing_gold_short():
    """Test GOLD short position sizing with 2% risk (high confidence)"""
    
    config = {
        "portfolio": {
            "target_risk_per_trade": 0.015,
            "max_risk_per_trade": 0.020,
            "aggressive_risk_threshold": 0.70,
        },
        "assets": {
            "GOLD": {
                "min_position_usd": 100,
                "max_position_usd": 6000,
                "risk": {
                    "min_stop_distance_pct": 0.008,
                    "max_stop_distance_pct": 0.03,
                }
            }
        }
    }
    
    portfolio_manager = Mock()
    portfolio_manager.current_capital = 10000
    portfolio_manager.positions = {}
    
    sizer = HybridPositionSizer(config, portfolio_manager)
    
    # Test: GOLD at $2700, stop at $2727 (1% away), high confidence
    entry_price = 2700
    stop_loss_price = 2727
    
    position_size, metadata = sizer.calculate_size_risk_based(
        asset="GOLD",
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        signal=-1,
        confidence_score=0.75,  # High confidence → 2% risk
    )
    
    # Verify
    expected_risk_usd = 10000 * 0.020  # $200 (high confidence)
    stop_distance_pct = (2727 - 2700) / 2700  # 0.01 (1%)
    expected_position = expected_risk_usd / stop_distance_pct  # $20,000
    
    # But clamped to max_position_usd = $6000
    assert position_size == 6000
    assert metadata["actual_risk_pct"] < 1.0  # Reduced due to max limit
    
    print(f"✅ Test passed (with max limit):")
    print(f"  Position: ${position_size:,.2f}")
    print(f"  Risk: ${metadata['actual_risk_usd']:,.2f} ({metadata['actual_risk_pct']:.2%})")


def test_stop_too_tight_rejection():
    """Test that positions with stops too tight are rejected"""
    
    config = {
        "portfolio": {"target_risk_per_trade": 0.015},
        "assets": {
            "BTC": {
                "min_position_usd": 100,
                "max_position_usd": 6000,
                "risk": {
                    "min_stop_distance_pct": 0.015,  # 1.5% minimum
                }
            }
        }
    }
    
    portfolio_manager = Mock()
    portfolio_manager.current_capital = 10000
    portfolio_manager.positions = {}
    
    sizer = HybridPositionSizer(config, portfolio_manager)
    
    # Test: Stop only 0.5% away (too tight!)
    entry_price = 95000
    stop_loss_price = 94525  # Only 0.5% away
    
    position_size, metadata = sizer.calculate_size_risk_based(
        asset="BTC",
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        signal=1,
    )
    
    # Should be rejected
    assert position_size == 0
    assert metadata["error"] == "stop_too_tight"
    assert metadata["stop_distance_pct"] < 0.015
    
    print(f"✅ Test passed: Stop too tight rejected")
    print(f"  Stop distance: {metadata['stop_distance_pct']:.2%}")
    print(f"  Minimum: 1.5%")


def test_portfolio_exposure_limits():
    """Test that portfolio exposure limits are respected"""
    
    config = {
        "portfolio": {
            "target_risk_per_trade": 0.015,
            "max_portfolio_exposure": 0.50,  # Only 50% can be used
        },
        "assets": {
            "BTC": {
                "min_position_usd": 100,
                "max_position_usd": 6000,
                "risk": {"min_stop_distance_pct": 0.01},
            }
        }
    }
    
    # Mock existing positions totaling $3,000
    existing_position = Mock()
    existing_position.quantity = 0.03
    existing_position.entry_price = 100000
    
    portfolio_manager = Mock()
    portfolio_manager.current_capital = 10000
    portfolio_manager.positions = {"pos1": existing_position}
    
    sizer = HybridPositionSizer(config, portfolio_manager)
    
    # Try to add position that would exceed 50%
    position_size, metadata = sizer.calculate_size_risk_based(
        asset="BTC",
        entry_price=95000,
        stop_loss_price=92000,
        signal=1,
    )
    
    # Should be clamped to fit within 50% total
    max_allowed = (10000 * 0.50) - 3000  # $5000 - $3000 = $2000
    assert position_size <= max_allowed
    
    print(f"✅ Test passed: Exposure limit respected")
    print(f"  Existing: $3,000")
    print(f"  New: ${position_size:,.2f}")
    print(f"  Total: ${3000 + position_size:,.2f} / $5,000 max")


# Run all tests
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RISK MANAGEMENT VERIFICATION TESTS")
    print("="*80 + "\n")
    
    test_risk_based_sizing_btc_long()
    print()
    test_risk_based_sizing_gold_short()
    print()
    test_stop_too_tight_rejection()
    print()
    test_portfolio_exposure_limits()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80 + "\n")