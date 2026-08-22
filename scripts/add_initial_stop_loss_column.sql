-- Dashboard: R-progress on open positions needs the FROZEN entry-time stop
-- (not the current, possibly-trailed one already in `stop_loss`) to compute
-- profit-in-R -- same distinction the R-batch/Fix-A R-lock logic already
-- makes internally (self.initial_stop_loss vs self.current_stop_loss).
-- Run once in: Supabase Dashboard -> SQL Editor.
-- Without this, src/database/database_manager.py's insert_trade_entry will
-- hit a PGRST204 "column not found" schema mismatch on every insert/update
-- that passes initial_stop_loss -- caught by the existing schema-mismatch
-- fallback (strips this field and retries), so the bot stays up either way,
-- but the column won't populate until this runs. Mirrors
-- add_livermore_state_columns.sql's pattern exactly.

ALTER TABLE trades
  ADD COLUMN IF NOT EXISTS initial_stop_loss numeric;
