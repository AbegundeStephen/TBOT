-- L1: Livermore state as top-level (indexed) columns on `trades`.
-- Run once in: Supabase Dashboard -> SQL Editor.
-- Without this, src/database/database_manager.py's insert_trade_entry will
-- hit a PGRST204 "column not found" schema mismatch on every insert/update
-- that passes livermore_state_4h/1h/age_4h/age_1h — caught by the existing
-- schema-mismatch fallback (strips these 4 fields and retries), so the bot
-- stays up either way, but the columns won't populate until this runs.

ALTER TABLE trades
  ADD COLUMN IF NOT EXISTS livermore_state_4h text,
  ADD COLUMN IF NOT EXISTS livermore_state_1h text,
  ADD COLUMN IF NOT EXISTS livermore_state_age_4h integer,
  ADD COLUMN IF NOT EXISTS livermore_state_age_1h integer;
