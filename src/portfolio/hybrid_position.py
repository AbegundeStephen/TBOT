"""
Hybrid Position Sizing System
Combines automated risk management with manual override capabilities
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SizingMode(Enum):
    """Position sizing modes"""
    AUTOMATED = "automated"      # Follow risk rules strictly
    MANUAL_OVERRIDE = "override"  # Manual size specified
    REDUCED_RISK = "reduced_risk" # Lower exposure for uncertain markets
    ELEVATED_RISK = "elevated"    # Higher exposure for high-conviction trades


class PositionSizingRequest:
    """Request object for position sizing with manual override support"""
    
    def __init__(
        self,
        asset: str,
        current_price: float,
        signal: int,
        mode: SizingMode = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        manual_size_pct: float = None,
        confidence_score: float = None,
        market_condition: str = None,  # "bullish", "bearish", "uncertain"
        override_reason: str = None,
        max_override_pct: float = 2.0,  # Prevent extreme overrides
    ):
        self.asset = asset
        self.current_price = current_price
        self.signal = signal
        self.mode = mode
        self.manual_size_usd = manual_size_usd
        self.manual_size_pct = manual_size_pct
        self.confidence_score = confidence_score or 0.5
        self.market_condition = market_condition or "neutral"
        self.override_reason = override_reason
        self.max_override_pct = max_override_pct


class HybridPositionSizer:
    """
    Hybrid position sizing with automated rules and manual overrides
    """
    
    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        
        # Override history for audit trail
        self.override_history = []
        
        logger.info("HybridPositionSizer initialized")
    
    def calculate_size(self, request: PositionSizingRequest) -> Tuple[float, Dict]:
        """
        Calculate position size with hybrid logic
        
        Returns:
            Tuple of (position_size_usd, metadata_dict)
        """
        try:
            # Step 1: Calculate base automated size
            base_size = self._calculate_automated_size(
                request.asset,
                request.current_price,
                request.signal
            )
            
            # Step 2: Apply confidence adjustments
            confidence_adjusted = self._apply_confidence_adjustment(
                base_size,
                request.confidence_score,
                request.market_condition
            )
            
            # Step 3: Apply manual override if requested
            if request.mode == SizingMode.MANUAL_OVERRIDE and request.manual_size_usd:
                final_size, override_result = self._apply_manual_override(
                    base_size,
                    confidence_adjusted,
                    request.manual_size_usd,
                    request.override_reason,
                    request.max_override_pct
                )
            elif request.mode == SizingMode.REDUCED_RISK:
                final_size = confidence_adjusted * 0.5
                override_result = {
                    "mode": "REDUCED_RISK",
                    "reason": "Lower exposure due to uncertain market conditions",
                    "reduction_pct": 50
                }
            elif request.mode == SizingMode.ELEVATED_RISK:
                final_size = min(
                    confidence_adjusted * 1.5,
                    self._get_max_position_size(request.asset)
                )
                override_result = {
                    "mode": "ELEVATED_RISK",
                    "reason": "Higher exposure for high-conviction trade",
                    "elevation_pct": 50
                }
            else:
                final_size = confidence_adjusted
                override_result = {"mode": "AUTOMATED", "applied_adjustments": []}
            
            # Step 4: Apply hard limits
            final_size = self._apply_hard_limits(
                request.asset,
                final_size,
                request.signal
            )
            
            # Build metadata
            metadata = {
                "asset": request.asset,
                "mode": request.mode.value,
                "signal": request.signal,
                "confidence_score": request.confidence_score,
                "market_condition": request.market_condition,
                "base_size_usd": base_size,
                "confidence_adjusted_usd": confidence_adjusted,
                "final_size_usd": final_size,
                "override_details": override_result,
                "timestamp": datetime.now().isoformat(),
                "portfolio_status": {
                    "available_capital": self.portfolio_manager.current_capital,
                    "current_exposure": self._calculate_current_exposure(),
                    "max_allowed": self._get_max_position_size(request.asset)
                }
            }
            
            # Log the decision
            self._log_sizing_decision(metadata)
            
            # Track overrides
            if request.mode != SizingMode.AUTOMATED:
                self.override_history.append(metadata)
            
            return final_size, metadata
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0, {"error": str(e)}
    
    def _calculate_automated_size(
        self,
        asset: str,
        current_price: float,
        signal: int
    ) -> float:
        """Calculate base position size using portfolio risk rules"""
        
        asset_cfg = self.config["assets"][asset]
        
        # Base: % of capital
        base_pct = self.portfolio_cfg.get("base_position_size", 0.10)
        base_size = self.portfolio_manager.current_capital * base_pct
        
        # Apply asset weight
        asset_weight = asset_cfg.get("weight", 1.0)
        base_size *= asset_weight
        
        # Apply signal confidence boost (optional)
        # BUY signals typically get standard sizing, SELL signals might get reduced
        if signal == -1:  # SELL signal
            base_size *= 0.8
        
        # Apply max risk per trade
        max_risk_pct = self.risk_cfg.get("max_risk_per_trade", 0.02)
        max_risk_usd = self.portfolio_manager.current_capital * max_risk_pct
        
        # Estimate SL distance to calculate max size
        risk_cfg = asset_cfg.get("risk", {})
        stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.02)
        sl_distance = current_price * stop_loss_pct
        
        # Max size based on risk
        max_size_by_risk = max_risk_usd / stop_loss_pct if stop_loss_pct > 0 else base_size
        base_size = min(base_size, max_size_by_risk)
        
        # Enforce min/max
        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        base_size = max(min_size, min(base_size, max_size))
        
        logger.debug(
            f"{asset}: Automated base size = ${base_size:,.2f} "
            f"(weight={asset_weight}, signal={signal})"
        )
        
        return base_size
    
    def _apply_confidence_adjustment(
        self,
        base_size: float,
        confidence_score: float,
        market_condition: str
    ) -> float:
        """Adjust size based on signal confidence and market conditions"""
        
        adjusted_size = base_size
        adjustments = []
        
        # Confidence-based scaling (0.3 to 1.3x)
        confidence_scalar = 0.5 + (confidence_score * 1.0)  # Maps [0,1] to [0.5,1.5]
        confidence_scalar = max(0.3, min(1.5, confidence_scalar))  # Clamp to [0.3, 1.5]
        adjusted_size *= confidence_scalar
        adjustments.append(f"confidence_scalar={confidence_scalar:.2f}")
        
        # Market condition adjustments
        condition_scalars = {
            "bullish": 1.1,
            "neutral": 1.0,
            "bearish": 0.8,
            "uncertain": 0.6,
            "extreme_volatility": 0.5
        }
        condition_scalar = condition_scalars.get(market_condition, 1.0)
        adjusted_size *= condition_scalar
        adjustments.append(f"market_condition={market_condition}_{condition_scalar:.2f}x")
        
        logger.debug(
            f"Confidence adjustment: {base_size:.2f} → {adjusted_size:.2f} "
            f"({', '.join(adjustments)})"
        )
        
        return adjusted_size
    
    def _apply_manual_override(
        self,
        base_size: float,
        confidence_adjusted: float,
        manual_size_usd: float,
        override_reason: str,
        max_override_pct: float
    ) -> Tuple[float, Dict]:
        """Apply manual override with safety guards"""
        
        # Validate override is within reasonable bounds
        min_allowed = confidence_adjusted * (1 - max_override_pct / 100)
        max_allowed = confidence_adjusted * (1 + max_override_pct / 100)
        
        if manual_size_usd < min_allowed or manual_size_usd > max_allowed:
            logger.warning(
                f"Manual override ${manual_size_usd:,.2f} exceeds bounds "
                f"[${min_allowed:,.2f}, ${max_allowed:,.2f}]. "
                f"Clamping to bounds."
            )
            manual_size_usd = max(min_allowed, min(manual_size_usd, max_allowed))
        
        deviation_pct = ((manual_size_usd - confidence_adjusted) / confidence_adjusted * 100) if confidence_adjusted > 0 else 0
        
        result = {
            "mode": "MANUAL_OVERRIDE",
            "reason": override_reason or "User override",
            "base_size": base_size,
            "confidence_adjusted": confidence_adjusted,
            "manual_size": manual_size_usd,
            "deviation_pct": deviation_pct,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Manual override applied: ${confidence_adjusted:,.2f} → ${manual_size_usd:,.2f} "
            f"({deviation_pct:+.1f}%) - Reason: {override_reason}"
        )
        
        return manual_size_usd, result
    
    def _apply_hard_limits(
        self,
        asset: str,
        position_size: float,
        signal: int
    ) -> float:
        """Apply absolute limits to prevent excessive exposure"""
        
        asset_cfg = self.config["assets"][asset]
        
        # Hard limits
        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        max_exposure = self.portfolio_cfg.get("max_portfolio_exposure", 0.95)
        max_single_asset = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)
        
        # Check absolute limits
        if position_size < min_size:
            logger.debug(f"Position size ${position_size:,.2f} below minimum ${min_size}, skipping")
            return 0.0
        
        position_size = min(position_size, max_size)
        
        # Check portfolio exposure
        current_exposure = self._calculate_current_exposure()
        max_portfolio_usd = self.portfolio_manager.current_capital * max_exposure
        if current_exposure + position_size > max_portfolio_usd:
            position_size = max(0, max_portfolio_usd - current_exposure)
            logger.warning(
                f"Position clamped to respect portfolio exposure limit: ${position_size:,.2f}"
            )
        
        # Check single asset limit
        max_asset_usd = self.portfolio_manager.current_capital * max_single_asset
        position_size = min(position_size, max_asset_usd)
        
        return position_size
    
    def _calculate_current_exposure(self) -> float:
        """Calculate total current portfolio exposure in USD"""
        return sum(
            pos.quantity * pos.entry_price
            for pos in self.portfolio_manager.positions.values()
        )
    
    def _get_max_position_size(self, asset: str) -> float:
        """Get maximum allowed position size for an asset"""
        asset_cfg = self.config["assets"][asset]
        max_usd = asset_cfg.get("max_position_usd", 6000)
        max_asset_pct = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)
        max_asset_usd = self.portfolio_manager.current_capital * max_asset_pct
        return min(max_usd, max_asset_usd)
    
    def _log_sizing_decision(self, metadata: Dict):
        """Log sizing decision for audit and analysis"""
        logger.info(
            f"[SIZING] {metadata['asset']} | Mode: {metadata['mode']} | "
            f"Confidence: {metadata['confidence_score']:.2f} | "
            f"Size: ${metadata['final_size_usd']:,.2f}"
        )
    
    def get_override_history(self, limit: int = 50) -> list:
        """Get recent override history"""
        return self.override_history[-limit:]
    
    def generate_sizing_report(self) -> Dict:
        """Generate summary report of sizing decisions"""
        if not self.override_history:
            return {"total_overrides": 0, "message": "No overrides applied"}
        
        automated_count = sum(1 for h in self.override_history if h["mode"] == "AUTOMATED")
        override_count = len(self.override_history) - automated_count
        
        override_reasons = {}
        for h in self.override_history:
            if h["mode"] != "AUTOMATED":
                reason = h.get("override_details", {}).get("reason", "Unknown")
                override_reasons[reason] = override_reasons.get(reason, 0) + 1
        
        avg_override_pct = sum(
            abs(h.get("override_details", {}).get("deviation_pct", 0))
            for h in self.override_history if h["mode"] != "AUTOMATED"
        ) / max(override_count, 1)
        
        return {
            "total_decisions": len(self.override_history),
            "automated": automated_count,
            "manual_overrides": override_count,
            "override_pct": (override_count / len(self.override_history) * 100) if self.override_history else 0,
            "avg_override_deviation_pct": avg_override_pct,
            "override_reasons": override_reasons
        }


# Example usage in execution handlers
class ExecutionHandlerWithHybridSizing:
    """Example showing how to integrate hybrid sizing into execution handlers"""
    
    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.sizer = HybridPositionSizer(config, portfolio_manager)
    
    def execute_with_hybrid_sizing(
        self,
        signal: int,
        current_price: float,
        asset: str = "BTC",
        confidence_score: float = None,
        market_condition: str = "neutral",
        sizing_mode: SizingMode = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None
    ) -> bool:
        """Execute trade with hybrid position sizing"""
        
        # Create sizing request
        request = PositionSizingRequest(
            asset=asset,
            current_price=current_price,
            signal=signal,
            mode=sizing_mode,
            manual_size_usd=manual_size_usd,
            confidence_score=confidence_score,
            market_condition=market_condition,
            override_reason=override_reason
        )
        
        # Calculate size
        position_size_usd, metadata = self.sizer.calculate_size(request)
        
        if position_size_usd <= 0:
            logger.warning(f"{asset}: Invalid position size calculated")
            return False
        
        # Proceed with position opening using the calculated size
        logger.info(
            f"[EXECUTE] {asset} - Opening ${position_size_usd:,.2f} position "
            f"(mode={sizing_mode.value}, confidence={confidence_score})"
        )
        
        # TODO: Call your normal _open_position or _open_mt5_position with size_usd parameter
        
        return True 