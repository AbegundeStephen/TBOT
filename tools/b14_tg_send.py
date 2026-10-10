"""B14 item 7.7 (Desire 8 Oct): send one Telegram message from a scheduled job (the Saturday job), the same way the
bot's own direct sender does: token and admin ids from .env (TELEGRAM_BOT_TOKEN / TELEGRAM_ADMIN_IDS), else from
config\\config.json. Prints what happened; exit code 1 if it could not be sent to anyone.
usage: python tools\\b14_tg_send.py "message text" """
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
text = " ".join(sys.argv[1:]).strip() or "TBOT: (empty message)"
tok = os.getenv("TELEGRAM_BOT_TOKEN")
ids = [i.strip() for i in (os.getenv("TELEGRAM_ADMIN_IDS") or "").split(",") if i.strip()]
if not tok or not ids:
    try:
        tg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig")).get("telegram", {}) or {}
        tok = tok or tg.get("bot_token")
        ids = ids or [str(i) for i in (tg.get("admin_ids") or [])]
    except Exception as e:
        print("config\\config.json could not be read:", e)
if not tok or not ids:
    print("TELEGRAM NOT SENT: no bot token / admin ids found")
    sys.exit(1)
import requests
sent = 0
for cid in ids:
    try:
        r = requests.post("https://api.telegram.org/bot%s/sendMessage" % tok, json={"chat_id": cid, "text": text},
                          timeout=10)
        if r.ok:
            sent += 1
        else:
            print("TELEGRAM NOT SENT to one admin: HTTP %s %s" % (r.status_code, r.text[:200]))
    except Exception as e:
        print("TELEGRAM NOT SENT to one admin:", e)
print("telegram sent to %d of %d admin(s)" % (sent, len(ids)))
sys.exit(0 if sent else 1)
