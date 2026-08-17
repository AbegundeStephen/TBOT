"""Standalone sanity check for TBOT_SHADOW_INSTRUMENT_FIX_SPEC.md segments 1-5,8.
Not part of the test suite -- ad hoc verification since backtest.py never
reaches shadow_trader.py's open_position/tick_update path."""
import sys
sys.path.insert(0, ".")

from src.execution.shadow_trader import ShadowTradingEngine, FRICTION_PENALTIES

engine = ShadowTradingEngine()

# S7h: atr=None must refuse the open
pos_none = engine.open_position(
    asset="BTC", side="long", entry_price=50000.0,
    strategy_source="TF", gate_blocked_by="HOLD (Score: 2.71/4.1)",
    signal_details={}, atr=None,
)
assert pos_none is None, "S7h FAILED: open_position should refuse when atr is None"
print("S7h OK: atr=None refused")

# S7a: mixed-case MT5 variant must resolve via the map, not the default
assert FRICTION_PENALTIES.get("XAUUSDM") == 0.0008, "S7a FAILED: XAUUSDm normalization"
print("S7a OK: normalized keys resolve", FRICTION_PENALTIES.get("XAUUSDM"))

# Full open -> BE trigger -> close cycle
pos = engine.open_position(
    asset="GOLD", side="long", entry_price=2000.0,
    strategy_source="TF", gate_blocked_by="HOLD (Score: 2.71/4.1)",
    signal_details={"setup_ref": 0}, atr=10.0, atr_multiplier=1.8,
    tp_multiples=[2.5, 4.0], trail_mult=0.8, be_r=0.75,
)
assert pos is not None
assert pos.gate_code == "HOLD", f"S7e FAILED: gate_code={pos.gate_code!r}"
assert pos.initial_stop_loss == pos.stop_loss and pos.initial_stop_loss != 0.0, "S7d FAILED: initial_stop_loss not frozen"
assert pos.trailing_distance == 10.0 * 0.8, f"S7c FAILED: trailing_distance={pos.trailing_distance}"
print(f"S7e/S7d/S7c OK at open: gate_code={pos.gate_code} initial_stop_loss={pos.initial_stop_loss} trailing_distance={pos.trailing_distance}")

risk = abs(pos.entry_price - pos.initial_stop_loss)
be_trigger_price = pos.entry_price + 0.75 * risk + 0.01
closed = pos.tick_update(be_trigger_price)
assert not closed
assert pos.breakeven_applied, "S7c FAILED: breakeven not applied at 0.75R"
# Trailing (activation threshold $10) is looser than this BE trigger ($13.51)
# at this ATR, so trailing legitimately fires in the same tick and can push
# the stop past entry -- the invariant is "no worse than breakeven", not "==".
assert pos.stop_loss >= pos.entry_price, "S7c FAILED: stop worse than breakeven"
print(f"S7c OK: breakeven applied at price={be_trigger_price}, stop_loss={pos.stop_loss} (>= entry {pos.entry_price})")

# Close it flat (protected-capital scratch scenario): gross tiny, net negative from friction
pos._close(pos.entry_price + 0.5, "manual_test_close")
d = pos.to_dict()
for k in ("brc_confirmed", "brc_kind", "stop_source", "initial_stop_loss",
          "friction_source", "trailing_activated", "net_pnl_r", "gate_code", "outcome_class"):
    assert k in d, f"S7d/S7e/S7b FAILED: {k} missing from to_dict"
print("S7d/S7b OK: to_dict has", {k: d[k] for k in ("outcome_class", "net_pnl_r", "friction_source", "gate_code")})

engine.closed_results.append(d)
# second record, different gate string but same code, to test bucket merge
pos2 = engine.open_position(
    asset="GOLD", side="short", entry_price=2000.0,
    strategy_source="TF", gate_blocked_by="HOLD (Score: 1.10/4.1)",
    signal_details={}, atr=10.0, tp_multiples=[2.5],
)
pos2._close(2050.0, "stop_loss")  # a real loss
engine.closed_results.append(pos2.to_dict())

sc = engine.get_gate_scorecard()
assert "HOLD" in sc, f"S7e FAILED: scorecard buckets={list(sc.keys())}"
assert sc["HOLD"]["count"] == 2, f"S7e FAILED: HOLD bucket count={sc['HOLD']['count']}"
assert "scratch_count" in sc["HOLD"] and "win_rate_ex_scratch" in sc["HOLD"], "S7b FAILED: scorecard missing scratch fields"
print("S7e/S7b OK: scorecard =", sc["HOLD"])

print("\nALL CHECKS PASSED")
