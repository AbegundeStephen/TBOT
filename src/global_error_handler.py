"""
Global Error Handler with Telegram Notifications
Catches all errors across the trading bot and sends intelligent notifications
"""

import logging
import traceback
import sys
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from functools import wraps
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""

    CRITICAL = "🔴 CRITICAL"
    ERROR = "🟠 ERROR"
    WARNING = "🟡 WARNING"
    INFO = "🟢 INFO"
    RECOVERY = "✅ RECOVERY"


@dataclass
class ErrorContext:
    """Context information for an error"""

    severity: ErrorSeverity
    component: str
    message: str
    exception: Optional[Exception]
    traceback_str: str
    timestamp: datetime
    additional_info: Dict[str, Any] = field(default_factory=dict)
    error_hash: str = ""

    def __post_init__(self):
        # Generate unique hash for deduplication
        error_signature = (
            f"{self.component}:{type(self.exception).__name__}:{self.message}"
        )
        self.error_hash = hashlib.md5(error_signature.encode()).hexdigest()


class ErrorAggregator:
    """
    Aggregates and deduplicates errors to prevent notification spam
    """

    def __init__(self, window_seconds: int = 300, max_duplicate_notifications: int = 3):
        self.window_seconds = window_seconds
        self.max_duplicate_notifications = max_duplicate_notifications

        # Track error occurrences: {error_hash: [timestamps]}
        self.error_occurrences: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Track notification count: {error_hash: count}
        self.notification_count: Dict[str, int] = defaultdict(int)

        # Track last notification time: {error_hash: timestamp}
        self.last_notification: Dict[str, datetime] = {}

    def should_notify(self, error_ctx: ErrorContext) -> tuple[bool, Optional[str]]:
        """
        Determine if we should send a notification for this error

        Returns:
            (should_notify: bool, aggregation_message: Optional[str])
        """
        now = datetime.now()
        error_hash = error_ctx.error_hash

        # Always notify CRITICAL errors
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            return True, None

        # Record this occurrence
        self.error_occurrences[error_hash].append(now)

        # Clean old occurrences outside window
        cutoff = now - timedelta(seconds=self.window_seconds)
        while (
            self.error_occurrences[error_hash]
            and self.error_occurrences[error_hash][0] < cutoff
        ):
            self.error_occurrences[error_hash].popleft()

        # Count occurrences in current window
        occurrence_count = len(self.error_occurrences[error_hash])

        # Check if we've exceeded notification limit
        if self.notification_count[error_hash] >= self.max_duplicate_notifications:
            last_notif = self.last_notification.get(error_hash)
            if last_notif and (now - last_notif).total_seconds() < self.window_seconds:
                # Suppress notification
                return (
                    False,
                    f"(Suppressed: {occurrence_count} occurrences in {self.window_seconds}s)",
                )

        # Send notification and update counters
        self.notification_count[error_hash] += 1
        self.last_notification[error_hash] = now

        # Include aggregation info if this is a repeat
        if occurrence_count > 1:
            return True, f"(Occurred {occurrence_count}x in {self.window_seconds}s)"

        return True, None

    def reset_error(self, error_hash: str):
        """Reset tracking for a specific error (e.g., after recovery)"""
        if error_hash in self.notification_count:
            del self.notification_count[error_hash]
        if error_hash in self.last_notification:
            del self.last_notification[error_hash]

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error occurrences in current window"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        summary = {}
        for error_hash, timestamps in self.error_occurrences.items():
            recent_count = sum(1 for ts in timestamps if ts >= cutoff)
            if recent_count > 0:
                summary[error_hash] = recent_count

        return summary


class GlobalErrorHandler:
    """
    Global error handler that catches all errors and sends Telegram notifications
    """

    def __init__(self, telegram_bot=None, db_manager=None, config: Dict = None):
        self.telegram_bot = telegram_bot
        self.db_manager = db_manager
        self.config = config or {}

        # Error aggregation
        self.aggregator = ErrorAggregator(
            window_seconds=self.config.get("error_window_seconds", 300),
            max_duplicate_notifications=self.config.get(
                "max_duplicate_notifications", 3
            ),
        )

        # Component-specific error tracking
        self.component_errors: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.component_error_counts: Dict[str, int] = defaultdict(int)

        # Recovery tracking
        self.recovering_components: set = set()

        logger.info("[ERROR HANDLER] Global error handler initialized")

    def handle_error(
        self,
        exception: Exception,
        component: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        additional_info: Dict = None,
        notify: bool = True,
    ):
        """
        Main error handling method

        Args:
            exception: The exception that occurred
            component: Component name (e.g., "binance_handler", "mt5_handler")
            severity: Error severity level
            additional_info: Additional context to include
            notify: Whether to send Telegram notification
        """
        try:
            # Create error context
            error_ctx = ErrorContext(
                severity=severity,
                component=component,
                message=str(exception),
                exception=exception,
                traceback_str=traceback.format_exc(),
                timestamp=datetime.now(),
                additional_info=additional_info or {},
            )

            # Track error
            self.component_errors[component].append(error_ctx)
            self.component_error_counts[component] += 1

            # Log error
            self._log_error(error_ctx)

            # Log to database if available
            if self.db_manager:
                self._log_to_database(error_ctx)

            # Send notification if needed
            if notify and self.telegram_bot:
                self._send_notification(error_ctx)

        except Exception as e:
            # Meta-error: error handler itself failed
            logger.error(f"[ERROR HANDLER] Failed to handle error: {e}", exc_info=True)

    def _log_error(self, error_ctx: ErrorContext):
        """Log error to console/file"""
        log_level = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.RECOVERY: logging.INFO,
        }.get(error_ctx.severity, logging.ERROR)

        logger.log(
            log_level,
            f"[{error_ctx.component.upper()}] {error_ctx.severity.value}: {error_ctx.message}",
        )

        # Log full traceback for errors and critical
        if error_ctx.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            logger.debug(f"Traceback:\n{error_ctx.traceback_str}")

    def _log_to_database(self, error_ctx: ErrorContext):
        """Log error to database"""
        try:
            self.db_manager.log_system_event(
                event_type="error",
                severity=error_ctx.severity.name.lower(),
                message=error_ctx.message,
                component=error_ctx.component,
                metadata={
                    "exception_type": type(error_ctx.exception).__name__,
                    "traceback": error_ctx.traceback_str,
                    "additional_info": error_ctx.additional_info,
                    "error_hash": error_ctx.error_hash,
                },
            )
        except Exception as e:
            logger.error(f"[ERROR HANDLER] Failed to log to database: {e}")

    def _send_notification(self, error_ctx: ErrorContext):
        """Send Telegram notification"""
        try:
            # Check if we should notify (deduplication)
            should_notify, aggregation_msg = self.aggregator.should_notify(error_ctx)

            if not should_notify:
                logger.debug(
                    f"[ERROR HANDLER] Suppressing duplicate notification for {error_ctx.component}"
                )
                return

            # Format message
            message = self._format_telegram_message(error_ctx, aggregation_msg)

            # Send via Telegram bot
            if hasattr(self.telegram_bot, "_send_telegram_notification"):
                self.telegram_bot._send_telegram_notification(
                    self.telegram_bot.send_notification(message, disable_preview=True)
                )
            elif hasattr(self.telegram_bot, "send_notification"):
                asyncio.create_task(self.telegram_bot.send_notification(message))
            else:
                logger.warning(
                    "[ERROR HANDLER] Telegram bot doesn't have notification method"
                )

        except Exception as e:
            logger.error(f"[ERROR HANDLER] Failed to send notification: {e}")

    def _format_telegram_message(
        self, error_ctx: ErrorContext, aggregation_msg: Optional[str]
    ) -> str:
        """Format error message for Telegram"""
        lines = [
            f"{error_ctx.severity.value} *ERROR ALERT*",
            "",
            f"*Component:* `{error_ctx.component}`",
            f"*Time:* {error_ctx.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"*Error:* {error_ctx.message[:200]}",  # Limit message length
        ]

        # Add aggregation info
        if aggregation_msg:
            lines.append(f"_{aggregation_msg}_")

        # Add exception type
        if error_ctx.exception:
            lines.append(f"*Type:* `{type(error_ctx.exception).__name__}`")

        # Add additional context
        if error_ctx.additional_info:
            lines.append("")
            lines.append("*Context:*")
            for key, value in list(error_ctx.additional_info.items())[
                :5
            ]:  # Limit to 5 items
                lines.append(f"  • {key}: `{str(value)[:50]}`")

        # Add traceback (truncated) for CRITICAL errors
        if error_ctx.severity == ErrorSeverity.CRITICAL and error_ctx.traceback_str:
            tb_lines = error_ctx.traceback_str.split("\n")
            last_lines = tb_lines[-5:]  # Last 5 lines of traceback
            lines.append("")
            lines.append("*Traceback (last 5 lines):*")
            lines.append("```")
            lines.extend(last_lines)
            lines.append("```")

        # Add component error summary
        error_count = self.component_error_counts.get(error_ctx.component, 0)
        if error_count > 1:
            lines.append("")
            lines.append(f"_Component error count: {error_count}_")

        return "\n".join(lines)

    def notify_recovery(self, component: str, message: str = "Component recovered"):
        """Send recovery notification"""
        try:
            if component in self.recovering_components:
                return  # Already notified

            self.recovering_components.add(component)

            recovery_msg = (
                f"{ErrorSeverity.RECOVERY.value} *RECOVERY*\n\n"
                f"*Component:* `{component}`\n"
                f"*Status:* {message}\n"
                f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"_System has recovered from previous errors_"
            )

            if self.telegram_bot:
                if hasattr(self.telegram_bot, "_send_telegram_notification"):
                    self.telegram_bot._send_telegram_notification(
                        self.telegram_bot.send_notification(recovery_msg)
                    )

            # Reset error tracking for this component
            if component in self.component_errors:
                for error_ctx in self.component_errors[component]:
                    self.aggregator.reset_error(error_ctx.error_hash)

            logger.info(f"[ERROR HANDLER] Recovery notification sent for {component}")

        except Exception as e:
            logger.error(f"[ERROR HANDLER] Failed to send recovery notification: {e}")

    def get_error_summary(self) -> Dict:
        """Get summary of all errors"""
        return {
            "component_error_counts": dict(self.component_error_counts),
            "recent_errors": {
                comp: [
                    {
                        "severity": err.severity.value,
                        "message": err.message,
                        "timestamp": err.timestamp.isoformat(),
                    }
                    for err in errors[-5:]  # Last 5 errors per component
                ]
                for comp, errors in self.component_errors.items()
            },
            "aggregator_summary": self.aggregator.get_error_summary(),
        }


# ====================================================================================
# DECORATOR FOR AUTOMATIC ERROR HANDLING
# ====================================================================================


def handle_errors(
    component: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    notify: bool = True,
    reraise: bool = False,
    default_return=None,
):
    """
    Decorator for automatic error handling

    Usage:
        @handle_errors("binance_handler", severity=ErrorSeverity.CRITICAL, notify=True)
        def execute_trade(self, ...):
            # Your code here
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler from args (assumes it's on self)
                error_handler = None
                if args and hasattr(args[0], "error_handler"):
                    error_handler = args[0].error_handler
                elif (
                    args
                    and hasattr(args[0], "trading_bot")
                    and hasattr(args[0].trading_bot, "error_handler")
                ):
                    error_handler = args[0].trading_bot.error_handler

                # Handle error if handler available
                if error_handler:
                    error_handler.handle_error(
                        exception=e,
                        component=component,
                        severity=severity,
                        additional_info={
                            "function": func.__name__,
                            "args": str(args[1:])[:100],  # Skip self
                        },
                        notify=notify,
                    )
                else:
                    logger.error(
                        f"[{component}] Error in {func.__name__}: {e}", exc_info=True
                    )

                # Reraise or return default
                if reraise:
                    raise
                return default_return

        # Async version
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_handler = None
                    if args and hasattr(args[0], "error_handler"):
                        error_handler = args[0].error_handler
                    elif (
                        args
                        and hasattr(args[0], "trading_bot")
                        and hasattr(args[0].trading_bot, "error_handler")
                    ):
                        error_handler = args[0].trading_bot.error_handler

                    if error_handler:
                        error_handler.handle_error(
                            exception=e,
                            component=component,
                            severity=severity,
                            additional_info={
                                "function": func.__name__,
                                "args": str(args[1:])[:100],
                            },
                            notify=notify,
                        )
                    else:
                        logger.error(
                            f"[{component}] Error in {func.__name__}: {e}",
                            exc_info=True,
                        )

                    if reraise:
                        raise
                    return default_return

            return async_wrapper

        return wrapper

    return decorator


# ====================================================================================
# CONTEXT MANAGER FOR ERROR HANDLING
# ====================================================================================


class ErrorContext:
    """Context manager for error handling in code blocks"""

    def __init__(
        self,
        error_handler: GlobalErrorHandler,
        component: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        notify: bool = True,
        reraise: bool = True,
    ):
        self.error_handler = error_handler
        self.component = component
        self.severity = severity
        self.notify = notify
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_error(
                exception=exc_val,
                component=self.component,
                severity=self.severity,
                notify=self.notify,
            )

            # Suppress exception if not reraising
            return not self.reraise

        return False
