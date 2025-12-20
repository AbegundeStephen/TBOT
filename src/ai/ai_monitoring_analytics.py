"""
AI Validator Monitoring & Analytics Tools
=========================================
Additional utilities for tracking and analyzing AI validator performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AIValidatorMonitor:
    """
    Real-time monitoring and analytics for AI validator
    """
    
    def __init__(self, validator):
        """
        Args:
            validator: EnhancedHybridSignalValidator instance
        """
        self.validator = validator
        self.performance_snapshots = []
        self.last_report_time = None
        self.report_interval = 3600  # 1 hour
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            str: Formatted report text
        """
        try:
            stats = self.validator.get_statistics()
            
            # Handle case where get_statistics() returns None
            if stats is None:
                logger.error("[AI MONITOR] get_statistics() returned None - no data yet")
                return self._generate_empty_report()
        except Exception as e:
            logger.error(f"[AI MONITOR] Error getting statistics: {e}")
            return self._generate_empty_report()
        
        report = []
        report.append("")
        report.append("=" * 70)
        report.append("AI VALIDATOR PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall stats
        report.append("[OVERALL STATISTICS]")
        report.append(f"  Total Checks:    {stats['total_checks']}")
        report.append(f"  Approved:        {stats['approved']} ({stats['approval_rate']})")
        report.append(f"  Rejected:        {stats['rejected']} ({stats['rejection_rate']})")
        report.append("")
        
        # Rejection breakdown
        report.append("[REJECTION BREAKDOWN]")
        breakdown = stats['rejection_breakdown']
        for reason, count in breakdown.items():
            pct = (count / max(stats['rejected'], 1)) * 100
            report.append(f"  {reason:20s}: {count:4d} ({pct:.1f}%)")
        report.append("")
        
        # Bypass stats
        report.append("[BYPASS STATISTICS]")
        bypasses = stats['bypasses']
        report.append(f"  Strong Signal:   {bypasses['strong_signal']}")
        report.append(f"  Circuit Breaker: {bypasses['circuit_breaker']}")
        report.append("")
        
        # Current thresholds
        report.append("[CURRENT THRESHOLDS]")
        thresholds = stats['current_thresholds']
        report.append(f"  S/R Distance:    {thresholds['sr_threshold']}")
        report.append(f"  Pattern Conf:    {thresholds['pattern_confidence']}")
        report.append(f"  Adjustments:     {stats['adaptive_adjustments']}")
        report.append("")
        
        # Circuit breaker status
        cb = stats['circuit_breaker']
        report.append("[CIRCUIT BREAKER]")
        report.append(f"  Status:          {'ACTIVE' if cb['active'] else 'INACTIVE'}")
        if cb['active']:
            report.append(f"  Cooldown:        {cb['cooldown']} signals")
        report.append("")
        
        # Top rejection reasons
        if 'top_rejection_reasons' in stats:
            report.append("[TOP REJECTION REASONS]")
            for reason, count in stats['top_rejection_reasons'].items():
                report.append(f"  {reason:30s}: {count:4d}")
            report.append("")
        
        # Per-strategy stats
        if 'per_strategy' in stats:
            report.append("[PER-STRATEGY PERFORMANCE]")
            for strategy, strat_stats in stats['per_strategy'].items():
                approval_rate = (strat_stats['approved'] / max(strat_stats['checks'], 1)) * 100
                report.append(f"  {strategy}:")
                report.append(f"    Checks:   {strat_stats['checks']}")
                report.append(f"    Approved: {strat_stats['approved']} ({approval_rate:.1f}%)")
                report.append(f"    Rejected: {strat_stats['rejected']}")
            report.append("")
        
        report.append("=" * 70)
        report.append("")
        
        return "\n".join(report)
    
    def _generate_empty_report(self) -> str:
        """Generate report when no data available"""
        report = []
        report.append("")
        report.append("=" * 70)
        report.append("AI VALIDATOR PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("[OVERALL STATISTICS]")
        report.append("  No data available yet.")
        report.append("")
        report.append("=" * 70)
        report.append("")
        return "\n".join(report)
    
    def analyze_rejection_patterns(self) -> Dict:
        """
        Analyze patterns in rejections to identify issues
        
        Returns:
            Dict: Analysis results
        """
        try:
            history = list(self.validator.validation_history)
            
            if len(history) < 10:
                return {
                    "message": "Insufficient data for analysis",
                    "current_validations": len(history),
                    "required": 10
                }
            
            df = pd.DataFrame(history)
        except Exception as e:
            logger.error(f"[AI MONITOR] Error in analyze_rejection_patterns: {e}")
            return {
                "error": str(e),
                "message": "Failed to analyze rejection patterns"
            }
        
        analysis = {
            "total_validations": len(df),
            "time_range": {
                "start": df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        
        # Result distribution
        result_counts = df['result'].value_counts()
        analysis['result_distribution'] = {
            result: {
                "count": int(count),
                "percentage": f"{(count / len(df)) * 100:.1f}%"
            }
            for result, count in result_counts.items()
        }
        
        # Rejection reasons
        rejected = df[df['result'] == 'rejected']
        if len(rejected) > 0:
            reason_counts = rejected['reason'].value_counts()
            analysis['rejection_reasons'] = {
                reason: {
                    "count": int(count),
                    "percentage": f"{(count / len(rejected)) * 100:.1f}%"
                }
                for reason, count in reason_counts.items()
            }
        
        # Strategy performance
        strategy_groups = df.groupby(['strategy', 'result']).size().unstack(fill_value=0)
        analysis['strategy_performance'] = {}
        
        for strategy in strategy_groups.index:
            approved = strategy_groups.loc[strategy].get('approved', 0)
            rejected = strategy_groups.loc[strategy].get('rejected', 0)
            total = approved + rejected
            
            analysis['strategy_performance'][strategy] = {
                "approved": int(approved),
                "rejected": int(rejected),
                "approval_rate": f"{(approved / max(total, 1)) * 100:.1f}%"
            }
        
        # Pattern confidence analysis
        approved = df[df['result'] == 'approved']
        if len(approved) > 0 and 'confidence' in approved.columns:
            analysis['pattern_confidence'] = {
                "approved_avg": f"{approved['confidence'].mean():.2%}",
                "approved_min": f"{approved['confidence'].min():.2%}",
                "approved_max": f"{approved['confidence'].max():.2%}",
            }
        
        # S/R distance analysis
        if 'sr_distance' in df.columns:
            approved_sr = approved[approved['sr_distance'].notna()]
            rejected_sr = rejected[rejected['sr_distance'].notna()]
            
            if len(approved_sr) > 0:
                analysis['sr_distance_approved'] = {
                    "avg": f"{approved_sr['sr_distance'].mean():.2f}%",
                    "median": f"{approved_sr['sr_distance'].median():.2f}%",
                }
            
            if len(rejected_sr) > 0:
                analysis['sr_distance_rejected'] = {
                    "avg": f"{rejected_sr['sr_distance'].mean():.2f}%",
                    "median": f"{rejected_sr['sr_distance'].median():.2f}%",
                }
        
        # Validation time analysis
        if 'validation_time_ms' in df.columns:
            times = df[df['validation_time_ms'].notna()]['validation_time_ms']
            if len(times) > 0:
                analysis['validation_time'] = {
                    "avg_ms": f"{times.mean():.1f}",
                    "median_ms": f"{times.median():.1f}",
                    "max_ms": f"{times.max():.1f}",
                }
        
        return analysis
    
    def check_health(self) -> Dict:
        """
        Perform health check and return status
        
        Returns:
            Dict: Health status
        """
        try:
            stats = self.validator.get_statistics()
            
            # Handle None or missing stats
            if stats is None:
                return {
                    "status": "initializing",
                    "warnings": ["Validator not yet active"],
                    "issues": [],
                    "message": "No validations performed yet"
                }
        except Exception as e:
            logger.error(f"[AI MONITOR] Error in check_health: {e}")
            return {
                "status": "error",
                "warnings": [],
                "issues": [f"Failed to get statistics: {str(e)}"],
            }
        
        health = {
            "status": "healthy",
            "warnings": [],
            "issues": [],
        }
        
        # Check rejection rate
        total = stats['total_checks']
        if total > 50:  # Need enough data
            rejection_rate = stats['rejected'] / total
            
            if rejection_rate > 0.80:
                health['issues'].append(
                    f"Very high rejection rate: {rejection_rate:.0%}"
                )
                health['status'] = "critical"
            elif rejection_rate > 0.65:
                health['warnings'].append(
                    f"High rejection rate: {rejection_rate:.0%}"
                )
                if health['status'] == "healthy":
                    health['status'] = "warning"
        
        # Check circuit breaker
        if stats['circuit_breaker']['active']:
            health['warnings'].append(
                f"Circuit breaker active (cooldown: {stats['circuit_breaker']['cooldown']})"
            )
            if health['status'] == "healthy":
                health['status'] = "warning"
        
        # Check if validator is actually being used
        if stats['total_checks'] == 0:
            health['warnings'].append("No validations performed yet")
        
        # Check bypass rate
        total_bypasses = (
            stats['bypasses']['strong_signal'] +
            stats['bypasses']['circuit_breaker']
        )
        if total > 0:
            bypass_rate = total_bypasses / total
            if bypass_rate > 0.50:
                health['warnings'].append(
                    f"High bypass rate: {bypass_rate:.0%}"
                )
                if health['status'] == "healthy":
                    health['status'] = "warning"
        
        # Check for pattern recognition issues
        breakdown = stats['rejection_breakdown']
        if breakdown['no_pattern'] > breakdown['no_sr_level'] * 2:
            health['warnings'].append(
                "Pattern recognition may need tuning (high 'no_pattern' rejections)"
            )
        
        return health
    
    def get_threshold_trend(self, lookback_minutes: int = 60) -> Dict:
        """
        Analyze threshold adjustment trends
        
        Args:
            lookback_minutes: How far back to analyze
            
        Returns:
            Dict: Threshold trends
        """
        history = list(self.validator.threshold_history)
        
        if len(history) < 2:
            return {"message": "Insufficient threshold history"}
        
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        recent = [h for h in history if h['timestamp'] > cutoff]
        
        if len(recent) < 2:
            return {"message": f"No threshold changes in last {lookback_minutes} minutes"}
        
        df = pd.DataFrame(recent)
        
        return {
            "adjustments": len(recent),
            "sr_threshold": {
                "current": f"{df['sr_threshold'].iloc[-1]:.2%}",
                "min": f"{df['sr_threshold'].min():.2%}",
                "max": f"{df['sr_threshold'].max():.2%}",
                "trend": self._calculate_trend(df['sr_threshold'].values)
            },
            "pattern_threshold": {
                "current": f"{df['pattern_threshold'].iloc[-1]:.0%}",
                "min": f"{df['pattern_threshold'].min():.0%}",
                "max": f"{df['pattern_threshold'].max():.0%}",
                "trend": self._calculate_trend(df['pattern_threshold'].values)
            },
            "volatility_avg": f"{df['volatility'].mean():.2%}",
            "regime_confidence_avg": f"{df['regime_confidence'].mean():.2f}",
        }
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < values.mean() * 0.05:  # Less than 5% change
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def log_periodic_report(self, force: bool = False):
        """
        Log performance report if interval has passed
        
        Args:
            force: Force report even if interval hasn't passed
        """
        try:
            now = datetime.now()
            
            if force or self.last_report_time is None or \
               (now - self.last_report_time).total_seconds() >= self.report_interval:
                
                report = self.generate_performance_report()
                logger.info(report)
                
                # Log health check
                health = self.check_health()
                
                # Only log warnings/errors if not in initializing state
                if health['status'] not in ["initializing", "healthy"]:
                    logger.warning(f"[AI VALIDATOR HEALTH] Status: {health['status'].upper()}")
                    for warning in health.get('warnings', []):
                        logger.warning(f"  ⚠️  {warning}")
                    for issue in health.get('issues', []):
                        logger.error(f"  ❌ {issue}")
                elif health['status'] == "initializing":
                    logger.debug(f"[AI VALIDATOR] Status: Initializing - {health.get('message', 'No data yet')}")
                
                self.last_report_time = now
                
        except Exception as e:
            logger.error(f"[AI MONITOR] Error in log_periodic_report: {e}", exc_info=True)


class AIValidatorTuner:
    """
    Automatic tuning recommendations for AI validator
    """
    
    def __init__(self, validator):
        """
        Args:
            validator: EnhancedHybridSignalValidator instance
        """
        self.validator = validator
    
    def analyze_and_recommend(self) -> Dict:
        """
        Analyze performance and recommend adjustments
        
        Returns:
            Dict: Recommendations
        """
        try:
            stats = self.validator.get_statistics()
            
            # Handle None stats
            if stats is None:
                return {
                    "message": "Validator not yet active - no data to analyze",
                    "current_validations": 0
                }
            
            history = list(self.validator.validation_history)
            
            if len(history) < 100:
                return {
                    "message": "Need at least 100 validations for tuning recommendations",
                    "current_validations": len(history)
                }
        except Exception as e:
            logger.error(f"[AI TUNER] Error in analyze_and_recommend: {e}")
            return {
                "error": str(e),
                "message": "Failed to generate recommendations"
            }
        
        recommendations = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "current_settings": {
                "sr_threshold": f"{self.validator.current_sr_threshold:.2%}",
                "pattern_threshold": f"{self.validator.current_pattern_threshold:.0%}",
            },
            "recommendations": []
        }
        
        df = pd.DataFrame(history)
        
        # Analyze rejection rate
        rejection_rate = (df['result'] == 'rejected').mean()
        
        if rejection_rate > 0.75:
            recommendations['recommendations'].append({
                "priority": "high",
                "issue": "Very high rejection rate",
                "current_value": f"{rejection_rate:.0%}",
                "suggestion": "Consider increasing base S/R threshold or decreasing pattern confidence requirement",
                "proposed_changes": {
                    "sr_threshold": f"+{self.validator.base_sr_threshold * 0.25:.3%}",
                    "pattern_threshold": f"-{self.validator.base_pattern_confidence * 0.10:.1%}"
                }
            })
        elif rejection_rate > 0.60:
            recommendations['recommendations'].append({
                "priority": "medium",
                "issue": "High rejection rate",
                "current_value": f"{rejection_rate:.0%}",
                "suggestion": "Slightly relax thresholds",
                "proposed_changes": {
                    "sr_threshold": f"+{self.validator.base_sr_threshold * 0.15:.3%}",
                }
            })
        
        # Analyze rejection reasons
        rejected = df[df['result'] == 'rejected']
        if len(rejected) > 0:
            reason_dist = rejected['reason'].value_counts(normalize=True)
            
            # Too many S/R rejections
            no_sr_rate = reason_dist.get('no_sr_level', 0)
            if no_sr_rate > 0.50:
                recommendations['recommendations'].append({
                    "priority": "high",
                    "issue": "Majority of rejections are S/R related",
                    "current_value": f"{no_sr_rate:.0%}",
                    "suggestion": "Increase S/R threshold to allow wider distance",
                    "proposed_changes": {
                        "sr_threshold": f"+{self.validator.base_sr_threshold * 0.30:.3%}"
                    }
                })
            
            # Too many pattern rejections
            pattern_reasons = [r for r in reason_dist.index if 'pattern' in r or 'confidence' in r]
            pattern_reject_rate = sum(reason_dist.get(r, 0) for r in pattern_reasons)
            
            if pattern_reject_rate > 0.50:
                recommendations['recommendations'].append({
                    "priority": "high",
                    "issue": "Majority of rejections are pattern related",
                    "current_value": f"{pattern_reject_rate:.0%}",
                    "suggestion": "Lower pattern confidence threshold",
                    "proposed_changes": {
                        "pattern_threshold": f"-{self.validator.base_pattern_confidence * 0.15:.1%}"
                    }
                })
        
        # Analyze bypass usage
        total = stats['total_checks']
        if total > 0:
            bypass_rate = (
                stats['bypasses']['strong_signal'] +
                stats['bypasses']['circuit_breaker']
            ) / total
            
            if bypass_rate > 0.40:
                recommendations['recommendations'].append({
                    "priority": "medium",
                    "issue": "High bypass rate indicates validator may be too strict",
                    "current_value": f"{bypass_rate:.0%}",
                    "suggestion": "Review and potentially relax validation criteria"
                })
        
        # Check if no recommendations
        if not recommendations['recommendations']:
            recommendations['recommendations'].append({
                "priority": "info",
                "message": "Current settings appear appropriate",
                "rejection_rate": f"{rejection_rate:.0%}",
                "suggestion": "Continue monitoring performance"
            })
        
        return recommendations


# Integration example for TradingBot main.py
def integrate_ai_monitoring(trading_bot):
    """
    Example of how to integrate monitoring into TradingBot
    """
    if hasattr(trading_bot, 'ai_validator') and trading_bot.ai_validator:
        # Create monitor
        trading_bot.ai_monitor = AIValidatorMonitor(trading_bot.ai_validator)
        
        # Schedule periodic reports (add to your schedule setup)
        import schedule
        schedule.every(1).hours.do(
            trading_bot.ai_monitor.log_periodic_report
        )
        
        logger.info("[AI MONITOR] Initialized with hourly reporting")
        
        # Optional: Create tuner for manual analysis
        trading_bot.ai_tuner = AIValidatorTuner(trading_bot.ai_validator)
        
        # You can call tuner when needed:
        # recommendations = trading_bot.ai_tuner.analyze_and_recommend()
        # logger.info(json.dumps(recommendations, indent=2))