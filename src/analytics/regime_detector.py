import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_regime(adx: float, volatility_ratio: float = 1.0) -> str:
    """
    Classifies market state based on trend strength and volatility.
    
    Rules:
    - TREND: ADX > 25 (Strong directional conviction)
    - RANGE: ADX < 20 (Low directional conviction, sideways chop)
    - NEUTRAL: 20 <= ADX <= 25 (Transition state)
    
    Volatility Adjustment:
    - If volatility is extreme (> 2.0x avg), defaults to TREND regardless of ADX.
    """
    try:
        # Extreme Volatility Override
        if volatility_ratio > 2.0:
            return "TREND"
            
        if adx > 25:
            return "TREND"
        elif adx < 20:
            return "RANGE"
        else:
            return "NEUTRAL"
            
    except Exception as e:
        logger.error(f"[REGIME] Error detecting regime: {e}")
        return "NEUTRAL"
