"""
Hedging Support Module
======================
Enables simultaneous long/short positions without closing opposites.
Non-invasive enhancement to portfolio manager.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class HedgingSupport:
    """
    Manages hedging logic for portfolio manager.
    Allows simultaneous long/short positions on same asset.
    """
    
    def __init__(self, enabled: bool = True, max_hedge_ratio: float = 1.0):
        """
        Args:
            enabled: Whether hedging is allowed
            max_hedge_ratio: Maximum ratio of short to long exposure (1.0 = equal)
        """
        self.enabled = enabled
        self.max_hedge_ratio = max_hedge_ratio
        
        logger.info(f"[HEDGING] Support initialized")
        logger.info(f"  Enabled: {self.enabled}")
        logger.info(f"  Max Hedge Ratio: {self.max_hedge_ratio:.0%}")
    
    def can_open_hedge_position(
        self, 
        asset_name: str,
        new_side: str,
        new_size_usd: float,
        existing_positions: List,
        max_positions_per_side: int = 3
    ) -> Tuple[bool, str]:
        """
        Check if a hedge position can be opened.
        
        Args:
            asset_name: Asset name (e.g., "BTC")
            new_side: "long" or "short"
            new_size_usd: Size of new position in USD
            existing_positions: List of Position objects for this asset
            max_positions_per_side: Maximum positions per side (default 3)
        
        Returns:
            (can_open, reason)
        """
        if not self.enabled:
            # Hedging disabled - use original logic
            opposite_side = "short" if new_side == "long" else "long"
            opposite_positions = [p for p in existing_positions if p.side == opposite_side]
            
            if opposite_positions:
                return False, f"Hedging disabled - have opposite {opposite_side} position"
            
            # Check per-side limit
            same_side_positions = [p for p in existing_positions if p.side == new_side]
            if len(same_side_positions) >= max_positions_per_side:
                return False, f"Max {new_side} positions reached ({max_positions_per_side})"
            
            return True, "OK"
        
        # ================================================================
        # HEDGING ENABLED - Different Logic
        # ================================================================
        
        # Separate positions by side
        long_positions = [p for p in existing_positions if p.side == "long"]
        short_positions = [p for p in existing_positions if p.side == "short"]
        
        # Check per-side position limit
        if new_side == "long":
            if len(long_positions) >= max_positions_per_side:
                return False, f"Max long positions reached ({max_positions_per_side})"
        else:
            if len(short_positions) >= max_positions_per_side:
                return False, f"Max short positions reached ({max_positions_per_side})"
        
        # Calculate current exposure
        long_exposure = sum(p.quantity * p.entry_price for p in long_positions)
        short_exposure = sum(p.quantity * p.entry_price for p in short_positions)
        
        # Calculate new exposure after opening position
        if new_side == "long":
            new_long_exposure = long_exposure + new_size_usd
            new_short_exposure = short_exposure
        else:
            new_long_exposure = long_exposure
            new_short_exposure = short_exposure + new_size_usd
        
        # Check hedge ratio limit
        if new_long_exposure > 0:
            hedge_ratio = new_short_exposure / new_long_exposure
        elif new_short_exposure > 0:
            hedge_ratio = new_long_exposure / new_short_exposure
        else:
            hedge_ratio = 0
        
        if hedge_ratio > self.max_hedge_ratio:
            return False, f"Hedge ratio {hedge_ratio:.0%} exceeds max {self.max_hedge_ratio:.0%}"
        
        logger.info(f"[HEDGING] {asset_name} hedge position allowed")
        logger.info(f"  New Side: {new_side.upper()}")
        logger.info(f"  Long Exposure:  ${new_long_exposure:,.2f} ({len(long_positions)} positions)")
        logger.info(f"  Short Exposure: ${new_short_exposure:,.2f} ({len(short_positions)} positions)")
        logger.info(f"  Hedge Ratio:    {hedge_ratio:.2%}")
        
        return True, "Hedge allowed"
    
    def get_hedging_summary(self, positions: Dict) -> Dict:
        """
        Get summary of current hedging status.
        
        Args:
            positions: Dict of {position_id: Position}
        
        Returns:
            Dict with hedging metrics
        """
        if not positions:
            return {
                'total_positions': 0,
                'assets_hedged': [],
                'net_exposure': 0,
                'gross_exposure': 0,
                'hedge_effectiveness': 0
            }
        
        # Group by asset
        by_asset = {}
        for pos in positions.values():
            if pos.asset not in by_asset:
                by_asset[pos.asset] = {'long': [], 'short': []}
            
            by_asset[pos.asset][pos.side].append(pos)
        
        # Calculate metrics
        total_long_exposure = 0
        total_short_exposure = 0
        assets_hedged = []
        
        for asset, sides in by_asset.items():
            long_exp = sum(p.quantity * p.entry_price for p in sides['long'])
            short_exp = sum(p.quantity * p.entry_price for p in sides['short'])
            
            total_long_exposure += long_exp
            total_short_exposure += short_exp
            
            # Asset is hedged if has both long and short
            if sides['long'] and sides['short']:
                assets_hedged.append({
                    'asset': asset,
                    'long_exposure': long_exp,
                    'short_exposure': short_exp,
                    'net_exposure': long_exp - short_exp,
                    'hedge_ratio': short_exp / long_exp if long_exp > 0 else 0
                })
        
        net_exposure = total_long_exposure - total_short_exposure
        gross_exposure = total_long_exposure + total_short_exposure
        
        # Hedge effectiveness: how much of gross exposure is hedged
        if gross_exposure > 0:
            hedge_effectiveness = 1 - (abs(net_exposure) / gross_exposure)
        else:
            hedge_effectiveness = 0
        
        return {
            'total_positions': len(positions),
            'long_positions': sum(len(s['long']) for s in by_asset.values()),
            'short_positions': sum(len(s['short']) for s in by_asset.values()),
            'total_long_exposure': total_long_exposure,
            'total_short_exposure': total_short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'hedge_effectiveness': hedge_effectiveness,
            'assets_hedged': assets_hedged,
            'num_assets_hedged': len(assets_hedged)
        }


def enable_hedging_for_portfolio(portfolio_manager, max_hedge_ratio: float = 1.0):
    """
    Enable hedging support for a portfolio manager WITHOUT modifying its code.
    
    Usage:
        portfolio_manager = PortfolioManager(...)
        enable_hedging_for_portfolio(portfolio_manager, max_hedge_ratio=1.0)
    """
    
    # Create hedging support instance
    hedging = HedgingSupport(enabled=True, max_hedge_ratio=max_hedge_ratio)
    
    # Store original can_open_position method
    original_method = portfolio_manager.can_open_position
    
    # Create enhanced version
    def enhanced_can_open_position(asset: str, side: str) -> Tuple[bool, str]:
        """Enhanced version that supports hedging"""
        
        # Get existing positions for this asset
        existing_positions = portfolio_manager.get_asset_positions(asset)
        
        # Calculate position size (estimate)
        # This is a simplified version - real implementation would get actual size
        current_price = 50000 if asset == "BTC" else 2000  # Placeholder
        position_size_usd = portfolio_manager.current_capital * 0.02  # 2% risk
        
        # Check with hedging support
        return hedging.can_open_hedge_position(
            asset_name=asset,
            new_side=side,
            new_size_usd=position_size_usd,
            existing_positions=existing_positions,
            max_positions_per_side=portfolio_manager.max_positions_per_asset
        )
    
    # Replace method
    portfolio_manager.can_open_position = enhanced_can_open_position
    
    # Add hedging support as attribute for status queries
    portfolio_manager.hedging_support = hedging
    
    logger.info(f"[HEDGING] ✅ Hedging enabled for portfolio")
    logger.info(f"  Max Hedge Ratio: {max_hedge_ratio:.0%}")
    
    return portfolio_manager


def log_hedging_status(portfolio_manager):
    """
    Log current hedging status to console.
    """
    if not hasattr(portfolio_manager, 'hedging_support'):
        logger.info("[HEDGING] Not enabled")
        return
    
    summary = portfolio_manager.hedging_support.get_hedging_summary(
        portfolio_manager.positions
    )
    
    logger.info(f"\n{'='*70}")
    logger.info("[HEDGING STATUS]")
    logger.info(f"{'='*70}")
    logger.info(f"Total Positions:    {summary['total_positions']}")
    logger.info(f"  Long:             {summary['long_positions']}")
    logger.info(f"  Short:            {summary['short_positions']}")
    logger.info(f"")
    logger.info(f"Exposure:")
    logger.info(f"  Long:             ${summary['total_long_exposure']:,.2f}")
    logger.info(f"  Short:            ${summary['total_short_exposure']:,.2f}")
    logger.info(f"  Net:              ${summary['net_exposure']:+,.2f}")
    logger.info(f"  Gross:            ${summary['gross_exposure']:,.2f}")
    logger.info(f"")
    logger.info(f"Hedge Effectiveness: {summary['hedge_effectiveness']:.1%}")
    logger.info(f"Assets Hedged:       {summary['num_assets_hedged']}")
    
    if summary['assets_hedged']:
        logger.info(f"\nHedged Assets:")
        for hedge in summary['assets_hedged']:
            logger.info(f"  {hedge['asset']}:")
            logger.info(f"    Long:  ${hedge['long_exposure']:,.2f}")
            logger.info(f"    Short: ${hedge['short_exposure']:,.2f}")
            logger.info(f"    Net:   ${hedge['net_exposure']:+,.2f}")
            logger.info(f"    Ratio: {hedge['hedge_ratio']:.2%}")
    
    logger.info(f"{'='*70}\n")