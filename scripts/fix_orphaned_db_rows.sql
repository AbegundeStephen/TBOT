-- Fix orphaned trade rows in Supabase
-- Run in: Supabase Dashboard → SQL Editor
-- Orphaned rows = status='open' but no matching live position
-- (bot crashed or restarted without closing them in the DB)

-- STEP 1: Preview — see what would be closed
SELECT
    id,
    asset,
    side,
    entry_price,
    position_size_usd,
    entry_time,
    status,
    mt5_ticket,
    binance_order_id,
    NOW() - entry_time::timestamptz AS age
FROM trades
WHERE status = 'open'
ORDER BY entry_time ASC;

-- STEP 2: Close all stale open rows older than 24 hours
-- (adjust the interval if needed; conservative default)
-- Run this only after reviewing Step 1 output.
UPDATE trades
SET
    status      = 'closed',
    exit_reason = 'closed_offline',
    exit_time   = NOW(),
    pnl         = 0.0,
    pnl_pct     = 0.0
WHERE
    status = 'open'
    AND entry_time::timestamptz < NOW() - INTERVAL '24 hours';

-- STEP 3: Verify cleanup
SELECT COUNT(*) AS remaining_open FROM trades WHERE status = 'open';
