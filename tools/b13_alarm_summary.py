"""B13 item 14 (Desire 5 Oct): the midnight alarm summary -- reads YESTERDAY's log lines (from every
logs\\trading_bot.log* file, so log rotation at midnight can't empty it) and sends one Telegram summary.
Run by Task Scheduler at 00:10 box time. Read-only: it only reads logs and sends one message."""
import glob, os, re, sys, datetime, collections, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
day = (datetime.date.today() - datetime.timedelta(days=int(sys.argv[1]) if len(sys.argv) > 1 else 1)).isoformat()
cnt, fails, errors = collections.Counter(), collections.Counter(), collections.Counter()
n = 0
for f in sorted(glob.glob(os.path.join("logs", "trading_bot.log*"))):
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith(day):
                    continue
                n += 1
                if "HEARTBEAT FAIL" in line or "[HEARTBEAT] FAIL" in line:
                    m = re.search(r"FAIL[:\]]?\s*([\w.\-]+)", line)
                    fails[m.group(1) if m else "?"] += 1
                if " - ERROR - " in line or " - CRITICAL - " in line:
                    m = re.search(r" - (?:ERROR|CRITICAL) - (\[[^\]]+\])", line)
                    errors[m.group(1) if m else "(no tag)"] += 1
                for tag in ("Traceback", "[NS-PROOF]", "[PKG-HANDOVER]", "[PKG-ENTER]", "[PKG-CANCEL]", "[DIAG-CONFIRM]",
                            "Trade Opened", "[TRADE_EVENT] {\"event\": \"ENTRY\"", "[TRADE_EVENT] {\"event\": \"EXIT\"",
                            "[SAFETY]", "[FRESHNESS]", "[STOP] SHUTTING DOWN", "[WATCHDOG] MT5 connection lost",
                            "[CIRCUIT BREAKER] Halted", "[HEALTH] System is UNHEALTHY"):
                    if tag in line:
                        cnt[tag] += 1
    except Exception as e:
        errors["(could not read %s: %s)" % (f, e)] += 1
msg = ["TBOT ALARM SUMMARY for %s (%d log lines read)" % (day, n)]
if n == 0:
    msg.append("NO LOG LINES FOUND FOR THAT DAY -- the bot may not have been running, or the log files moved.")
msg.append("Heartbeat fails: " + (", ".join("%s x%d" % kv for kv in fails.most_common(12)) if fails else "none"))
msg.append("Errors by tag: " + (", ".join("%s x%d" % kv for kv in errors.most_common(12)) if errors else "none"))
msg.append("Events: " + (", ".join("%s x%d" % kv for kv in cnt.most_common()) if cnt else "none"))
text = "\n".join(msg)
print(text)
try:
    env = {}
    for l in open(".env", encoding="utf-8", errors="replace"):
        if "=" in l and not l.strip().startswith("#"):
            k, v = l.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    tok, ids = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_ADMIN_IDS", "")
    if tok and ids:
        import urllib.request
        for cid in [x.strip() for x in ids.split(",") if x.strip()]:
            req = urllib.request.Request("https://api.telegram.org/bot%s/sendMessage" % tok,
                                         data=json.dumps({"chat_id": cid, "text": text[:3900]}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read()
        print("sent to Telegram")
    else:
        print("NOT SENT: TELEGRAM_BOT_TOKEN / TELEGRAM_ADMIN_IDS not found in .env")
except Exception as e:
    print("NOT SENT:", e)
