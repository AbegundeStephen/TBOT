-- Fix orphaned trade rows in Supabase
-- Run in: Supabase Dashboard → SQL Editor
-- Orphaned rows = status='open' but no matching live position
-- (bot crashed or restarted without closing them in the DB)

-- STEP 1: Preview — see what would be closed
SELECT
    id, asset, side, entry_price, position_size_usd, entry_time,
    status, mt5_ticket, binance_order_id,
    NOW() - entry_time::timestamptz AS age
FROM trades
WHERE status = 'open'
ORDER BY entry_time ASC;

-- STEP 2: For each row above, set the REAL exit price and reason before
-- running anything — do not blank these blind.
--
-- Example for one specific row (id=123, known SL hit at 1.90769):
-- UPDATE trades
-- SET
--     status      = 'closed',
--     exit_reason = 'sl_offline_confirmed',
--     exit_time   = NOW(),
--     exit_price  = 1.90769,
--     pnl         = (1.90769 - entry_price) * quantity * (CASE WHEN side='short' THEN -1 ELSE 1 END),
--     pnl_pct     = ((1.90769 - entry_price) / entry_price) * 100 * (CASE WHEN side='short' THEN -1 ELSE 1 END)
-- WHERE id = 123;

-- STEP 3: Only for rows where the real outcome is genuinely unknown and
-- unrecoverable — close at entry price as documented "unknown, assumed
-- flat" rather than a silent fake zero.
-- UPDATE trades
-- SET
--     status      = 'closed',
--     exit_reason = 'closed_offline_unknown_outcome',
--     exit_time   = NOW(),
--     exit_price  = entry_price,
--     pnl         = 0.0,
--     pnl_pct     = 0.0
-- WHERE
--     status = 'open'
--     AND entry_time::timestamptz < NOW() - INTERVAL '24 hours'
--     AND id IN (/* explicit list of confirmed-unrecoverable IDs */);

-- STEP 4: Verify cleanup
SELECT COUNT(*) AS remaining_open FROM trades WHERE status = 'open';
