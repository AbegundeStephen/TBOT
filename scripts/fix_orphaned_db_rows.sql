-- Fix 12: Orphaned DB rows — run in Supabase SQL Editor
-- =========================================================
-- Identifies and optionally soft-deletes trade rows in the
-- 'trades' table that have no matching close event (exit_time IS NULL)
-- but are older than 24 hours — meaning the bot crashed or
-- was restarted without recording the close.
--
-- ALWAYS run the SELECT first to review before running DELETE/UPDATE.
--
-- Column mapping (adjust if your schema differs):
--   id          INTEGER PRIMARY KEY
--   asset       TEXT
--   side        TEXT
--   entry_price NUMERIC
--   exit_price  NUMERIC   (NULL if still open)
--   exit_time   TIMESTAMP (NULL if still open) — NOT 'closed_at'
--   pnl_pct     NUMERIC   (NULL if still open)
--   opened_at   TIMESTAMP
-- =========================================================

-- ── Step 1: INSPECT — review candidate orphan rows ────────────────────────────
SELECT
    id,
    asset,
    side,
    entry_price,
    exit_price,
    exit_time,
    pnl_pct,
    opened_at,
    NOW() - opened_at AS age
FROM trades
WHERE
    exit_time  IS NULL          -- no recorded close
    AND opened_at < NOW() - INTERVAL '24 hours'  -- older than one day
ORDER BY opened_at ASC;


-- ── Step 2: CLOSE orphans at NULL exit_price (marks them as force-closed) ─────
-- Uncomment and run ONLY after reviewing Step 1 output.
-- This sets exit_time to now and pnl_pct to 0 so the row is no longer "open".
--
-- UPDATE trades
-- SET
--     exit_time  = NOW(),
--     exit_price = entry_price,   -- recorded at entry (no fill data available)
--     pnl_pct    = 0.0
-- WHERE
--     exit_time  IS NULL
--     AND opened_at < NOW() - INTERVAL '24 hours';


-- ── Step 3 (optional): DELETE instead of soft-close ───────────────────────────
-- Use this ONLY if the rows are truly garbage (test trades, paper mode leftovers).
-- Uncomment and run ONLY after reviewing Step 1 output.
--
-- DELETE FROM trades
-- WHERE
--     exit_time  IS NULL
--     AND opened_at < NOW() - INTERVAL '24 hours';


-- ── Step 4: Verify — should return 0 rows after Step 2 or 3 ─────────────────
-- SELECT COUNT(*) AS remaining_orphans
-- FROM trades
-- WHERE exit_time IS NULL AND opened_at < NOW() - INTERVAL '24 hours';
