import pytest
from unittest.mock import Mock, patch
from src.execution.binance_handler import BinanceExecutionHandler
from src.market.price_cache import price_cache


@pytest.fixture
def mock_binance_handler_paper_mode():
    """Fixture to create a BinanceExecutionHandler in paper mode."""
    mock_config = {
        "trading": {"mode": "paper"},
        "assets": {"BTC": {"symbol": "BTCUSDT", "enable_futures": True}},
        "portfolio": {},
        "risk_management": {}  # Added to satisfy the BinanceExecutionHandler init
    }
    
    mock_client = Mock()
    mock_portfolio_manager = Mock()

    # Mock client.futures_exchange_info() for BinanceFuturesHandler initialization
    mock_client.futures_exchange_info.return_value = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                ],
            }
        ]
    }
    # Mock client.futures_account() for BinanceFuturesHandler initialization
    mock_client.futures_account.return_value = {}
    
    # Mock client.futures_change_position_mode() to prevent API calls
    mock_client.futures_change_position_mode.return_value = None


    # Ensure futures_handler is also mocked as it's checked in get_current_price
    with patch('src.execution.binance_futures.BinanceFuturesHandler') as MockFuturesHandler:
        mock_futures_instance = MockFuturesHandler.return_value
        
        # We need to set the mock_futures_instance.client to our mock_client
        # so that when BinanceFuturesHandler uses self.client in its methods,
        # it refers to our mocked client.
        mock_futures_instance.client = mock_client 

        # We also need to mock the _round_quantity method that is called
        # from within BinanceExecutionHandler's _open_position
        mock_futures_instance._round_quantity.side_effect = lambda qty: round(qty, 3)

        handler = BinanceExecutionHandler(mock_config, mock_client, mock_portfolio_manager)
        handler.futures_handler = mock_futures_instance # Manually set the mocked futures_handler
        yield handler


@pytest.fixture(autouse=True)
def clear_price_cache_for_each_test():
    """Fixture to clear the price_cache before each test."""
    price_cache._prices = {}
    price_cache._timestamps = {}


def test_get_current_price_paper_mode_returns_mock_price(mock_binance_handler_paper_mode):
    """
    Test that get_current_price returns a mock price when in paper mode
    and force_live is True, and does not call _fetch_live_futures_price.
    """
    handler = mock_binance_handler_paper_mode

    # Patch the internal _fetch_live_futures_price to ensure it's not called
    with patch.object(handler, '_fetch_live_futures_price') as mock_fetch_live:
        # Call get_current_price with force_live=True
        price = handler.get_current_price(symbol="BTCUSDT", force_live=True)

        # Assertions
        assert price == 40000.0  # Expect the hardcoded mock price
        mock_fetch_live.assert_not_called()  # Ensure live fetch was not attempted
        assert price_cache.get("BTCUSDT") == 40000.0 # Ensure mock price was cached


def test_get_current_price_paper_mode_cached_price_first(mock_binance_handler_paper_mode):
    """
    Test that get_current_price prioritizes cached price even in paper mode
    when force_live is True, and does not call _fetch_live_futures_price.
    """
    handler = mock_binance_handler_paper_mode

    # Prime the cache with a different price
    price_cache.set("BTCUSDT", 45000.0)

    with patch.object(handler, '_fetch_live_futures_price') as mock_fetch_live:
        price = handler.get_current_price(symbol="BTCUSDT", force_live=True)

        assert price == 45000.0  # Expect the primed cache price
        mock_fetch_live.assert_not_called()  # Ensure live fetch was not attempted

def test_get_current_price_paper_mode_no_force_live_cached_fallback(mock_binance_handler_paper_mode):
    """
    Test that get_current_price returns last known cached price when in paper mode
    and force_live is False, and no fresh price is available.
    """
    handler = mock_binance_handler_paper_mode

    price_cache.set("BTCUSDT", 39000.0) # Set a last known stale price

    with patch.object(handler, '_fetch_live_futures_price') as mock_fetch_live:
        price = handler.get_current_price(symbol="BTCUSDT", force_live=False)

        assert price == 39000.0 # Expect the last known stale price
        mock_fetch_live.assert_not_called() # Ensure live fetch was not attempted