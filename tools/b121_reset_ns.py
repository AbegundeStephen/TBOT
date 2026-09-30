"""B12.1 (decision 43 A) -- run ONCE from C:\\TradingBot\\TBOT with the bot STOPPED (a running bot saves over it).
Clears the new engine's saved memory -- and the practice lane's -- from every market's saved builder state, so on the
next start the engine rebuilds its setups from the last 30 days of CORRECTED candles (setups drawn from bad candles
before the 35 A fix are gone). Everything else in the saved state is kept. A .pre_B121 copy of every file is made
first. Setups found while catching up are logged, never traded; open positions are not touched."""
import glob
import os
import pickle
import shutil
import sys
sys.path.insert(0, os.getcwd())                     # the saved state holds project objects: load it from the repo root
files = sorted(glob.glob(os.path.join("data", "builder_state*", "*.pkl")))
if not files:
    print("no saved builder state found under data\\builder_state* -- nothing to do")
    sys.exit(0)
for f in files:
    try:
        with open(f, "rb") as fh:
            p = pickle.load(fh)
    except Exception as e:
        print("%s: could not be read (%s) -- left alone, tell Claude" % (f, e))
        continue
    if not isinstance(p, dict):
        print("%s: not the expected format -- left alone, tell Claude" % f)
        continue
    gone = [k for k in ("_ns_state", "_ns_x_state") if k in p]
    if not gone:
        print("%s: no saved engine memory -- nothing to clear" % f)
        continue
    shutil.copy2(f, f + ".pre_B121")
    for k in gone:
        p.pop(k)
    tmp = f + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(p, fh)
    os.replace(tmp, f)
    print("%s: cleared %s (backup: %s)" % (f, " and ".join(gone), os.path.basename(f) + ".pre_B121"))
print("done -- start the bot; each market's engine rebuilds from 30 days of corrected candles on its first cycle")
