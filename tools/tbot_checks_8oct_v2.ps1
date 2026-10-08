# =============================================================================================
#  TBOT CHECKS v2, 8 OCT  --  READ-ONLY: changes nothing, safe while the bot is running. A few minutes.
#  Replaces the first tbot_checks_8oct.ps1 (same checks, two of them tidied, plus the 25-day window).
#  Writes three files for Claude:
#     C:\TradingBot\TBOT\b13_final_output.txt   B13 step 20, now reading every rotated daily log
#     C:\TradingBot\TBOT\usoil_0704_check.txt   why no orders were tried 6 Oct 18:11 -> 7 Oct 10:42 (the USOIL
#                                               buy), plus the EURJPY 6 Oct and GOLD 7 Oct log lines
#     C:\TradingBot\TBOT\window_25d_check.txt   the engine's 25-day window: the code, what the bot fetched, and
#                                               what 60 days of candles would have changed
#  HOW TO RUN (Stephen):
#     1. Save this file as  C:\TradingBot\TBOT\tools\tbot_checks_8oct_v2.ps1
#     2. Open PowerShell AS ADMINISTRATOR (so the task list and process lines are complete), then:
#           cd C:\TradingBot\TBOT
#           powershell -ExecutionPolicy Bypass -File tools\tbot_checks_8oct_v2.ps1
#     3. It ends with three lines starting "WRITTEN:". Send all three .txt files to Desire.
#  Do not start it between 23:50 and 00:10 (the bot's log changes to a new file at midnight).
#  STOP and send a screenshot if the three "WRITTEN:" lines are not there at the end.
# =============================================================================================
param([string]$Root = "C:\TradingBot\TBOT", [string]$Py = "")
$ErrorActionPreference = "Continue"
Set-Location $Root
if (-not $Py) { $Py = Join-Path $Root "venv\Scripts\python.exe" }
$outA = Join-Path $Root "b13_final_output.txt"
$outB = Join-Path $Root "usoil_0704_check.txt"
$outC = Join-Path $Root "window_25d_check.txt"
"Working... (part A of 3)"

# ---------------------------------------------------------------------------------------------
# PART A -- b13_final_output.txt  (step 20 of the B13 doc; the logs now rotate at midnight,
#           so every search reads all daily files written since the B13 restart, oldest first)
# ---------------------------------------------------------------------------------------------
$restart = [datetime]"2026-10-07 10:42:00"                       # the B13 restart, box clock
$since   = @(Get-ChildItem logs\trading_bot.log* | Where-Object { $_.LastWriteTime -ge $restart } | Sort-Object LastWriteTime)
$recent  = @(Get-ChildItem logs\trading_bot.log* | Where-Object { $_.LastWriteTime -ge [datetime]"2026-10-05" } | Sort-Object LastWriteTime)
$first   = @{}                                                    # first line at or after the restart, per file
foreach ($f in $since) {
    $h = Select-String -LiteralPath $f.FullName -Pattern "^2026-10-07 (10:4[2-9]|10:5\d|1[1-9]:|2[0-3]:)" -List
    if ($h) { $first[$f.FullName] = $h.LineNumber } else { $first[$f.FullName] = 1 }
}
filter AfterRestart { if ($_.LineNumber -ge $first[$_.Path]) { $_ } }
$tags = "COMBINED-CHART|CYCLE-TIME|PKG-ROUTE|DIAG-CONFIRM|PKG-HANDOVER|PKG-ENTER|PKG-CANCEL|FRESHNESS|PROOF-SPENT|HEALTH\]|CIRCUIT BREAKER|WATCHDOG\] MT5|STOP\] SHUTTING|TRADE_EVENT"
& {
"== 0. box clock now $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | log files read since the B13 restart: $(($since | ForEach-Object { $_.Name }) -join ', ') =="
"== 1. installed files =="; & $Py -c "import hashlib;[print('%-45s %s' % (f, hashlib.sha256(open(f,'rb').read().replace(b'\r\n',b'\n')).hexdigest()[:16].upper())) for f in (r'main.py', r'src\execution\ns_engine.py', r'src\execution\ns_package.py', r'src\execution\mt5_handler.py', r'src\monitoring\health_monitor.py', r'src\execution\veteran_trade_manager.py', r'src\execution\composite_state_builder.py', r'src\telegram\__init__.py', r'src\execution\heartbeat.py', r'src\ai\ns_card.py')]"
"== 2. errors since restart =="; $since | Select-String -Pattern "Traceback|NameError|AttributeError|SyntaxError" | AfterRestart | Select-Object -Last 10
"-- ERROR / CRITICAL lines since restart (last 15) --"; $since | Select-String -Pattern " - (ERROR|CRITICAL) - " | AfterRestart | Select-Object -Last 15
"== 3. B13 lines =="; Get-ChildItem logs\charts\*_combined.png | Select-Object Name, LastWriteTime
$b13 = @($since | Select-String -Pattern $tags | AfterRestart)
"-- how many of each since restart --"; $b13 | ForEach-Object { if ($_.Line -match "($tags)") { $Matches[1] } } | Group-Object | Sort-Object Count -Descending | Format-Table Count, Name -AutoSize
"-- the last 40 --"; $b13 | Select-Object -Last 40
"== 4. watchdog check list =="; $since | Select-String -Pattern "HEARTBEAT" | AfterRestart | Select-Object -Last 15
"== 5. scheduled tasks =="; Get-ScheduledTask | Where-Object { $_.TaskPath -notlike "\Microsoft*" } | Select-Object TaskName, State, @{n="Action";e={($_.Actions.Execute + " " + $_.Actions.Arguments)}}, @{n="RunsAs";e={$_.Principal.UserId}} | Format-List
"== 6. for the items still open (15, 16, 25, 26) =="
"-- ny_open (logs since 5 Oct) --"; $recent | Select-String -Pattern "ny_open" | Select-Object -Last 5
"-- Bad Gateway (since restart) --"; $since | Select-String -Pattern "Bad Gateway" | AfterRestart | Select-Object -Last 5
"-- PERSIST restored (bars passed format) --"; $since | Select-String -Pattern "\[PERSIST\] .* restored" | AfterRestart | Select-Object -Last 3
"-- price files re-fetched --"; $since | Select-String -Pattern "stale|too old|re-fetch|refetch|outdated" | AfterRestart | Select-Object -Last 8
"-- where ny_open / the old midnight summary live --"; Get-ChildItem -Path $Root -Recurse -Include *.py,*.ps1,*.bat,*.json -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike "*venv*" -and $_.FullName -notlike "*backup*" -and $_.FullName -ne $PSCommandPath -and $_.Name -notlike "tbot_checks_8oct*" } | Select-String -Pattern "ny_open|alarm summary|ALARM SUMMARY" -List | Select-Object Path, LineNumber, Line
"== 7. notes from steps 11 and 16 (type them here if any) =="
} 2>&1 | Out-File -FilePath $outA -Encoding utf8 -Width 400
"WRITTEN: $outA"

"Working... (part B of 3)"

# ---------------------------------------------------------------------------------------------
# PART B -- usoil_0704_check.txt  (the reading is done by the short Python below, which only
#           reads logs, logs\gates, config and data\raw -- it writes nothing)
# ---------------------------------------------------------------------------------------------
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "tbot_usoil_0704_check.py"
@'
# usoil_0704_check (Part B) -- READ-ONLY: reads logs, logs\gates, config, data. Writes nothing.
import os, re, sys, json, glob, datetime as dt
if len(sys.argv) > 1: os.chdir(sys.argv[1])
def p(s=''): print(str(s).encode('ascii', 'replace').decode('ascii'))
def cut(s, n=320):
    s = str(s).rstrip('\r\n'); return s if len(s) <= n else s[:n] + ' ...'
def mtime(f): return dt.datetime.fromtimestamp(os.path.getmtime(f))
TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - (.+?) - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - (.*)')
MAIN = ('__main__', 'main')
def logs_since(day, pat='trading_bot.log*'):
    fs = [f for f in glob.glob(os.path.join('logs', pat)) if os.path.isfile(f) and mtime(f) >= day]
    return sorted(fs, key=os.path.getmtime)
def opener(f):
    with open(f, 'rb') as fh: bom = fh.read(2)
    return open(f, encoding='utf-16' if bom in (b'\xff\xfe', b'\xfe\xff') else 'utf-8', errors='replace')
def scan(files):        # (file, time, logger, level, message, raw); lines without a time inherit the last one
    for f in files:
        last = ''
        with opener(f) as fh:
            for raw in fh:
                m = TS.match(raw)
                if m: last = m.group(1); yield os.path.basename(f), last, m.group(2), m.group(3), m.group(4), raw
                else: yield os.path.basename(f), last, None, None, raw.strip(), raw
def noise(s): return (not s.strip()) or set(s.strip()) <= set('-=')
def key(n, l): return n is not None and (n in MAIN or n.startswith('src.portfolio') or n.startswith('src.global_error_handler') or l in ('WARNING', 'ERROR', 'CRITICAL'))
def show(rows, cap, title):
    p(title + '   (%d line(s)%s)' % (len(rows), ', first %d shown' % cap if len(rows) > cap else ''))
    for fn, raw in rows[:cap]: p('   ' + cut(raw))

COLS = [('TRADE-ASSET', lambda n, l, s: '[TRADE ASSET] Processing' in s), ('BREAKER', lambda n, l, s: '[CIRCUIT BREAKER]' in s),
        ('LIMIT', lambda n, l, s: '[LIMIT]' in s), ('TRADE-ERR', lambda n, l, s: '[TRADE_ASSET]' in s or '[trade_asset]' in s or 'trade failed' in s),
        ('MAIN-ALL', lambda n, l, s: n in MAIN), ('ERRORS', lambda n, l, s: l in ('ERROR', 'CRITICAL')),
        ('USOIL-CNCL', lambda n, l, s: '[COUNCIL-ENTRY] USOIL' in s), ('BTC-CNCL', lambda n, l, s: '[COUNCIL-ENTRY] BTC' in s),
        ('USOIL-MTF', lambda n, l, s: 'Updated AI context for USOIL' in s)]
TAGS5 = ('[NS-PROOF]', '[COUNCIL-ADVISORY]', '[COUNCIL-CP2]', '[SHADOW] Opened', '[ENTRY-WINDOW]', '[SAFETY]', '[LIMIT]', '[COOLDOWN]', '[SESSION]', '[MARKET]',
         '[SKIP]', '[TRADE ASSET]', '[PAPER-MARKET]', '[NS-SIZE]', '[FRESHNESS]', '[MT5]', '[SIZING]', '[COUNCIL-RESIT]', '[PRESEND]', 'order')
GTAGS = ('[NS-PROOF]', '[PKG-', '[NS-EXPLORE]', '[ENTRY-WINDOW]', '[FRESHNESS]', '[MT5]', '[COUNCIL-ADVISORY]', '[SIZING]', '[LIMIT]', '[COOLDOWN]', '[SAFETY]', '[NS] ')
RX_MISC = re.compile(r'CRITICAL ALERT|\[RESET\] Daily counters|\[SESSION\] Trading session started|Peak equity reset|/resume|/reset_equity|\[BREAKER\]|\[ALERT\]|override')
RX_EW = re.compile(r'\[ENTRY-WINDOW\]|Outside preferred session|Rollover Dead Zone|Session-open cooldown|\[COOLDOWN\]|\[MARKET\]')
C = dict(hours={}, cb=[], cbr={}, misc=[], w4=[], w5=[], ew=[], ewc={}, eur=[], gold=[], g48=[], gwin=[], gwin2=[])
NOISE = ('[LSM-RETIRE]', '[Livermore] Created', '[CSV]', '[PERSIST]', '[SYNC]', '[MTF AI]', '[MTF DB]', '[MTF REGIME]', '[ZONE-1D]', '[ANGLE]',
         '[S2-STRUCTURE]', '[M1-SWING]', '[M1-CONVERGE]', '[SIGNAL]', '[LIFECYCLE', '[TRANSITION]', '[EMA CONFIRMER]', '[FRICTION]', '[H2-SYMMETRY]',
         '[CEILING]', '[THRESHOLD-CAP]', '[BAR]', '[GATE-ID-TRACE]', '[ACTIVITY]', 'Fetching ', '[VTM-LADDER]', '[COUNCIL-CP1]', '[COUNCIL GATE]')
def quiet(s): return not any(x in s for x in NOISE)

def sec1():
    p('== 1. log files in logs\\ (newest last) ==')
    for f in sorted(glob.glob(os.path.join('logs', '*.log*')), key=os.path.getmtime)[-40:]:
        p('   %-34s %8.1f MB   last write %s' % (os.path.basename(f), os.path.getsize(f) / 1e6, mtime(f).strftime('%Y-%m-%d %H:%M')))
def collect():
    F = logs_since(dt.datetime(2026, 10, 5))
    p('   read for 5-7 Oct: ' + ', '.join(os.path.basename(f) for f in F))
    for fn, t, n, l, s, raw in scan(F):
        if t[:10] not in ('2026-10-05', '2026-10-06', '2026-10-07'): continue
        h = t[:13]
        if '2026-10-05 12' <= h <= '2026-10-07 12' and n is not None:
            row = C['hours'].setdefault(h, [0] * len(COLS))
            for i, (c, fx) in enumerate(COLS):
                if fx(n, l, s): row[i] += 1
        if n is not None and '[CIRCUIT BREAKER]' in s:
            C['cb'].append((fn, raw)); k = re.sub(r'[\d.]+', '#', s.split('[CIRCUIT BREAKER]', 1)[1])[:110]
            e = C['cbr'].setdefault(k, [0, t, t]); e[0] += 1; e[2] = t
        elif n is not None and '[LIMIT]' not in s and RX_MISC.search(s): C['misc'].append((fn, raw))
        if '2026-10-06 18:04:00' <= t < '2026-10-06 18:15:00' and key(n, l) and not noise(s): C['w4'].append((fn, raw))
        if '2026-10-07 04:03:00' <= t < '2026-10-07 04:05:00' and not noise(s) and (key(n, l) or ('USOIL' in s and any(x in s for x in TAGS5))): C['w5'].append((fn, raw))
        if n is not None and RX_EW.search(s):
            C['ew'].append((fn, raw)); a = re.search(r'\] ?([A-Z][A-Z0-9]{2,7})\b', s); k = (RX_EW.search(s).group(0), a.group(1) if a else '?')
            C['ewc'][k] = C['ewc'].get(k, 0) + 1
        if '2026-10-06 13:55:00' <= t < '2026-10-06 14:16:00' and not noise(s) and quiet(s) and (key(n, l) or 'EURJPY' in s) and not ('[TRADE ASSET] Processing' in s and 'EURJPY' not in s): C['eur'].append((fn, raw))
        if t[:10] == '2026-10-07' and n is not None and 'GOLD' in s and any(x in s for x in GTAGS): C['gold'].append((fn, raw))
        if '4256.8' in s: C['g48'].append((fn, raw))
        if not noise(s) and quiet(s) and (key(n, l) or 'GOLD' in s) and not ('[TRADE ASSET] Processing' in s and 'GOLD' not in s):
            if '2026-10-07 14:55:00' <= t < '2026-10-07 15:16:00': C['gwin'].append((fn, raw))
            elif '2026-10-07 17:25:00' <= t < '2026-10-07 17:46:00': C['gwin2'].append((fn, raw))
def sec2():
    p('== 2. did the bot try to trade? lines per box-clock hour, 5 Oct 12:00 -> 7 Oct 12:59 ==')
    p('   hour             ' + ' '.join('%11s' % c for c, _ in COLS))
    for h in sorted(C['hours']): p('   %s:00  ' % h + ' '.join('%11d' % v for v in C['hours'][h]))
def sec3():
    p('== 3. circuit breaker, 5-7 Oct (trading_bot.log) ==')
    p('   [CIRCUIT BREAKER] lines: %d' % len(C['cb']))
    for k, (c, a, b) in sorted(C['cbr'].items(), key=lambda x: x[1][1]): p('   %6d x  first %s  last %s  |%s' % (c, a, b, k))
    cb = C['cb']
    for fn, raw in (cb[:3] + cb[-3:] if len(cb) > 6 else cb): p('   ' + fn + ': ' + cut(raw))
    show(C['misc'], 40, '-- daily reset / session start / peak reset / resume / alert lines --')
def sec3b():
    F = logs_since(dt.datetime(2026, 10, 6), 'bot_*.log*')
    p('-- 3b. the console copy (bot_*.log, warnings only) for the same days: ' + (', '.join(os.path.basename(f) for f in F) or 'none found'))
    cnt, first, last = {}, {}, {}
    for fn, t, n, l, s, raw in scan(F):
        if t[:10] not in ('2026-10-06', '2026-10-07'): continue
        for tag in ('[CIRCUIT BREAKER]', '[LIMIT]', 'Traceback', 'Logging error'):
            if tag in raw:
                cnt[tag] = cnt.get(tag, 0) + 1; first.setdefault(tag, cut(raw, 200)); last[tag] = cut(raw, 200)
    for tag in cnt: p('   %-18s %6d x | first: %s\n%s| last:  %s' % (tag, cnt[tag], first[tag], ' ' * 30, last[tag]))
def sec4(): show(C['w4'], 220, '== 4. 6 Oct 18:04-18:14 box (the second pass stops here): every main.py, portfolio and warning/error line ==')
def sec5(): show(C['w5'], 220, '== 5. 7 Oct 04:03-04:04 box (the USOIL buy proof): main.py, portfolio, warnings + USOIL decision lines ==')
def sec6():
    p('== 6. entry-window / session / market lines, 5-7 Oct ==')
    for k in sorted(C['ewc']): p('   %6d x  %s  %s' % (C['ewc'][k], k[0], k[1]))
    for fn, raw in C['ew'][-10:]: p('   ' + cut(raw))
def sec7():
    p('== 7. gate ledger (logs\\gates), all markets ==')
    W = [('6 Oct 08:00-18:05', '2026-10-06T08:00', '2026-10-06T18:06'), ('6 Oct 18:11-7 Oct 10:42', '2026-10-06T18:11', '2026-10-07T10:42'), ('7 Oct 10:43-23:59', '2026-10-07T10:43', '2026-10-08T00:00')]
    gc, ep = {}, []
    for gf in ('gates_2026-10-06.jsonl', 'gates_2026-10-07.jsonl'):
        fp = os.path.join('logs', 'gates', gf)
        if not os.path.exists(fp): p('   missing: ' + fp); continue
        with open(fp, encoding='utf-8', errors='replace') as fh:
            for raw in fh:
                try: r = json.loads(raw)
                except Exception: continue
                ts = str(r.get('ts', ''))
                for i, (nm, a, b) in enumerate(W):
                    if a <= ts < b: gc.setdefault((str(r.get('gate')), str(r.get('verdict'))), [0, 0, 0])[i] += 1
                if r.get('asset') == 'USOIL' and ('2026-10-07T04:00' <= ts < '2026-10-07T04:10' or '2026-10-06T09:00' <= ts < '2026-10-06T09:06'): ep.append(r)
    p('   %-26s %-10s %s' % ('gate', 'verdict', ' | '.join(w[0] for w in W)))
    for k in sorted(gc): p('   %-26s %-10s %17d %25d %19d' % (k[0], k[1], gc[k][0], gc[k][1], gc[k][2]))
    p('   USOIL rows 6 Oct 09:00-09:05 (the sell that traded) and 7 Oct 04:00-04:09 (the buy that did not):')
    for r in ep: p('   %s  %-34s %-20s %-9s %s' % (str(r.get('ts'))[:19], r.get('episode_id'), r.get('gate'), r.get('verdict'), cut(json.dumps(r.get('numbers'), default=str), 150)))
def sec8():
    p('== 8. are main.py lines reaching the log? per file since 1 Oct ==')
    p('   %-30s %9s %9s %9s %9s %9s %9s' % ('file', 'lines', 'main', 'main-warn', 'TRADE-ASS', 'SIGNALS', 'data_mgr'))
    for f in logs_since(dt.datetime(2026, 10, 1)):
        c = [0] * 6
        with opener(f) as fh:
            for raw in fh:
                c[0] += 1; m = TS.match(raw)
                if not m: continue
                n, l, s = m.group(2), m.group(3), m.group(4)
                if n in MAIN: c[1] += 1; c[2] += l != 'INFO'; c[3] += '[TRADE ASSET] Processing' in s; c[4] += '[SIGNALS] Updating signals' in s
                elif n == 'src.data.data_manager': c[5] += 1
        p('   %-30s %9d %9d %9d %9d %9d %9d' % ((os.path.basename(f),) + tuple(c)))
def sec9():
    p('== 9. settings as saved on disk (the bot reads them once, when it starts) ==')
    env = os.environ.get('TBOT_CONFIG_PATH')
    KEYS = [('trading', 'session_filter_enabled'), ('trading', 'startup_quarantine_minutes'), ('trading', 'mode'), ('risk_management', 'max_daily_loss_pct'),
            ('risk_management', 'max_loss_streak'), ('risk_management', 'circuit_breaker_loss_pct'), ('risk_management', 'max_daily_trades'), ('portfolio', 'max_drawdown'),
            ('portfolio', 'profit_lock_threshold'), ('logging', 'level'), ('logging', 'file'), ('phase_config', 'council_advisory'), ('phase_config', 'council_suspended'),
            ('phase_config', 'ns_paper_markets')]
    for cf in [x for x in [env, os.path.join('config', 'config.json'), os.path.join('config', 'config.prod.json')] if x]:
        if not os.path.exists(cf): p('   %s: not found' % cf); continue
        try: c = json.load(open(cf, encoding='utf-8-sig'))
        except Exception as e: p('   %s: cannot read (%s)' % (cf, e)); continue
        p('   %s  (last write %s)' % (cf, mtime(cf).strftime('%Y-%m-%d %H:%M')))
        for a, b in KEYS: p('      %s.%s = %s' % (a, b, json.dumps((c.get(a) or {}).get(b, '<not set>'))))
    for sf in (os.path.join('data', 'council_trial.json'), os.path.join('data', 'portfolio_state.pkl')):
        if os.path.exists(sf):
            p('   %s  last write %s%s' % (sf, mtime(sf).strftime('%Y-%m-%d %H:%M'), ('  ' + cut(open(sf, encoding='utf-8', errors='replace').read(), 300)) if sf.endswith('.json') else ''))
def load(base):
    fs = sorted(glob.glob(os.path.join('data', 'raw', base + '*_1h.csv')))
    if not fs: return None, os.path.join('data', 'raw', base + '*_1h.csv')
    rows = []
    with open(fs[0], encoding='utf-8-sig', errors='replace') as fh:
        head = [h.strip().lower() for h in fh.readline().split(',')]
        ti = next((head.index(c) for c in ('timestamp', 'time', 'datetime', 'date', 'open_time') if c in head), 0)
        ix = {c: head.index(c) for c in ('high', 'low', 'close') if c in head}
        for raw in fh:
            v = raw.strip().split(',')
            try:
                ts = v[ti]
                if re.match(r'^\d+(\.\d+)?$', ts): ts = dt.datetime.utcfromtimestamp(float(ts) / (1000 if float(ts) > 1e11 else 1)).strftime('%Y-%m-%d %H:%M:%S')
                rows.append((dt.datetime.strptime(ts[:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S'), float(v[ix['high']]), float(v[ix['low']]), float(v[ix['close']])))
            except Exception: continue
    return rows, fs[0]
def sec10():
    p('== 10. how old were the lines? (the live engine only gets about 25 days of hourly candles) ==')
    CHECKS = [('USOIL', 'USOIL sell 6 Oct 07:00 - old low by close', 'close', 87.805, 0.0006, '2026-10-06 07:00'),
              ('USOIL', 'USOIL sell 6 Oct 07:00 - old low by wick', 'low', 87.836, 0.0006, '2026-10-06 07:00'),
              ('USOIL', 'USOIL buy 7 Oct 02:00 - old high by close', 'close', 89.767, 0.0006, '2026-10-07 02:00'),
              ('USOIL', 'USOIL buy 7 Oct 02:00 - old high by wick', 'high', 89.952, 0.0006, '2026-10-07 02:00'),
              ('EURJPY', 'EURJPY buy 6 Oct 12:00 - its line 177.664', 'any', 177.664, 0.0006, '2026-10-06 12:00'),
              ('XAUUSD', 'GOLD sell 7 Oct 13:00 - its line 4256.82', 'any', 4256.82, 0.006, '2026-10-07 13:00')]
    cache = {}
    for base, what, col, lv, tol, sig in CHECKS:
        if base not in cache: cache[base] = load(base)
        rows, fp = cache[base]
        if rows is None: p('   %s: price file not found (%s)' % (what, fp)); continue
        s0 = dt.datetime.strptime(sig, '%Y-%m-%d %H:%M'); hits = set()
        for t, hi, lo, cl in rows:
            if not (s0 - dt.timedelta(days=150) <= t < s0): continue
            vals = {'high': hi, 'low': lo, 'close': cl}
            for c in (('high', 'low', 'close') if col == 'any' else (col,)):
                if abs(vals[c] - lv) <= tol: hits.add((t, c, vals[c]))
        hits = sorted(hits)[-4:]
        p('   %s  [%s]%s' % (what, os.path.basename(fp), '' if hits else ': no candle within %g of it in the 150 days before' % tol))
        for t, c, v in hits:
            age = (s0 - t).total_seconds() / 86400.0
            p('      %s  %-5s %-12g %6.1f days before the signal -> %s' % (t.strftime('%Y-%m-%d %H:%M'), c, v, age, 'OUTSIDE the live 25 days' if age > 24.5 else ('borderline' if age > 23 else 'inside')))
def sec11(): show(C['eur'], 200, '== 11. EURJPY 6 Oct 13:55-14:15 box (the rules bought at 12:00 UTC; MT5 did not): EURJPY + main.py + warning lines ==')
def sec12():
    show(C['gold'], 80, '== 12a. GOLD 7 Oct: engine / package / sending lines ==')
    show(C['g48'], 30, '-- 12b. any line with the GOLD line 4256.8 --')
    show(C['gwin'], 150, '-- 12c. GOLD 7 Oct 14:55-15:15 box (the 13:00 UTC signal): GOLD + main.py + warning lines --')
    show(C['gwin2'], 150, '-- 12d. GOLD 7 Oct 17:25-17:45 box (the 15:30 UTC package entry): GOLD + main.py + warning lines --')
for name, fx in [('1', sec1), ('scan', collect), ('2', sec2), ('3', sec3), ('3b', sec3b), ('4', sec4), ('5', sec5), ('6', sec6), ('7', sec7),
                 ('8', sec8), ('9', sec9), ('10', sec10), ('11', sec11), ('12', sec12)]:
    try:
        fx(); p()
    except Exception as e:
        p('   !! part %s stopped: %r' % (name, e)); p()
p('END OF usoil_0704_check')
'@ | Set-Content -Path $tmp -Encoding UTF8
& {
"== 0. how the bot is running (box clock now $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) =="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { "   python pid {0}  started {1}  {2}" -f $_.ProcessId, $_.CreationDate, $_.CommandLine }
foreach ($v in "TBOT_CONFIG_PATH", "TBOT_INSTANCE") { "   {0}: this window='{1}'  user='{2}'  machine='{3}'" -f $v, [Environment]::GetEnvironmentVariable($v, "Process"), [Environment]::GetEnvironmentVariable($v, "User"), [Environment]::GetEnvironmentVariable($v, "Machine") }
Get-ScheduledTask | Where-Object { $_.TaskPath -notlike "\Microsoft*" } | ForEach-Object { $t = $_; foreach ($a in $t.Actions) { "   task {0} [{1}] runs as {2}: {3} {4} (start in {5})" -f $t.TaskName, $t.State, $t.Principal.UserId, $a.Execute, $a.Arguments, $a.WorkingDirectory } }
"-- launcher lines (TBOT_ settings, python, main.py, Tee, config) --"
Get-ChildItem -Path "$Root\*.ps1", "$Root\*.bat", "$Root\scripts\*.ps1", "$Root\scripts\*.bat" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -ne $PSCommandPath -and $_.Name -notlike "tbot_checks_8oct*" } | Select-String -Pattern "TBOT_|python|main\.py|Tee-Object|config\.json" | ForEach-Object { "   {0}:{1}: {2}" -f $_.Filename, $_.LineNumber, $_.Line.Trim() }
""
& $Py $tmp $Root
} 2>&1 | Out-File -FilePath $outB -Encoding utf8 -Width 400
Remove-Item $tmp -ErrorAction SilentlyContinue
"WRITTEN: $outB"
"Working... (part C of 3)"

# ---------------------------------------------------------------------------------------------
# PART C -- window_25d_check.txt  (reads main.py, src, config, logs and data\raw; loads its own copy
#           of the engine code in this window only -- the running bot is not touched)
# ---------------------------------------------------------------------------------------------
$tmpC = Join-Path ([System.IO.Path]::GetTempPath()) "tbot_window_25d_check.py"
@'
# window_25d_check (Part C) -- READ-ONLY: reads main.py, src, config, logs and data\raw. Writes nothing.
import os, re, sys, json, glob, datetime as dt, importlib.util, logging
if len(sys.argv) > 1: os.chdir(sys.argv[1])
sys.path.insert(0, os.getcwd())
logging.getLogger().addHandler(logging.NullHandler())
def p(s=''): print(str(s).encode('ascii', 'replace').decode('ascii'))
def cut(s, n=200):
    s = str(s).rstrip('\r\n'); return s if len(s) <= n else s[:n] + ' ...'
def mtime(f): return dt.datetime.fromtimestamp(os.path.getmtime(f))
def logs_since(day):
    fs = [f for f in glob.glob(os.path.join('logs', 'trading_bot.log*')) if os.path.isfile(f) and mtime(f) >= day]
    return sorted(fs, key=os.path.getmtime)
CFG = {}
for _cf in [os.environ.get('TBOT_CONFIG_PATH'), os.path.join('config', 'config.json')]:
    if _cf and os.path.exists(_cf):
        try: CFG = json.load(open(_cf, encoding='utf-8-sig')); break
        except Exception: pass
PC = CFG.get('phase_config') or {}
M = None

def secC1():
    p('== C1. the installed code: what the engine is given, and how far back it looks ==')
    def show(path, pats, defs=None):
        if not os.path.exists(path): p('   %s: not found' % path); return
        cur = ''
        for i, l in enumerate(open(path, encoding='utf-8', errors='replace').read().splitlines(), 1):
            m = re.match(r'\s*def (\w+)', l)
            if m: cur = m.group(1)
            if (defs is None or cur in defs) and any(re.search(x, l) for x in pats):
                p('   %s:%d: %s%s' % (os.path.basename(path), i, l.strip()[:150], ('   [in %s]' % cur) if defs else ''))
    show('main.py', [r'lookback\s*=\s*\d+', r'timedelta\(days='], defs={'trade_asset', '_update_asset_signal', '_fetch_4h_data', '_fetch_1d_data'})
    show(os.path.join('src', 'execution', 'composite_state_builder.py'), [r'_ns_engine\.update\(', r'_ns_engine = NSEngine\('])
    show(os.path.join('src', 'execution', 'ns_engine.py'), [r'^(PKG_AHEAD_DAYS|PKG_NEAR|REPLAY_DAYS|BRAIN_HIST_DAYS|LAYER1_DAYS|LAYER2_DAYS|STALE_DAYS|DIAG_LIFE|DIAG_RECENT_H|AHEAD_ON_WICKS|FRESH_ATR|K)\s*=',
                                                            r'df4 = hourly_to_4h\(df1\)', r'len\(df1\) < ', r'no old 4H high/low ahead in the last'])

def secC2():
    p('== C2. what the live bot actually fetched (latest per market, logs of the last 2 days) ==')
    RX_F = re.compile(r'Fetching (\S+) (H1|H4|D1|W1) from MT5: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S* to (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
    RX_B = re.compile(r'\[M1-SWING\] (\w+): bars=(\d+)')
    fetch, bars = {}, {}
    for f in logs_since(dt.datetime.now() - dt.timedelta(days=2)):
        with open(f, encoding='utf-8', errors='replace') as fh:
            for raw in fh:
                if 'Fetching ' in raw:
                    m = RX_F.search(raw)
                    if m: fetch[(m.group(1), m.group(2))] = (m.group(3), m.group(4), raw[:19])
                elif '[M1-SWING]' in raw:
                    m = RX_B.search(raw)
                    if m:
                        b = bars.setdefault(m.group(1), [10 ** 9, 0, 0]); n = int(m.group(2)); b[0] = min(b[0], n); b[1] = max(b[1], n); b[2] = n
    if not fetch: p('   no "Fetching ... from MT5" lines found')
    for (sym, tf) in sorted(fetch):
        a, b, at = fetch[(sym, tf)]
        try: span = (dt.datetime.strptime(b, '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')).total_seconds() / 86400
        except Exception: span = float('nan')
        p('   %-9s %s  %s -> %s  = %5.1f days   (logged %s)' % (sym, tf, a, b, span, at))
    for a in sorted(bars): p('   [M1-SWING] %-7s hourly bars the engine got: latest %d (lowest %d, highest %d)' % (a, bars[a][2], bars[a][0], bars[a][1]))

def secC3():
    p('== C3. per-market engine switches in the config the bot reads (phase_config.ns_markets) ==')
    nm = PC.get('ns_markets') or {}
    if not nm: p('   phase_config.ns_markets not found -- the engine then uses its built-in defaults')
    for a in sorted(nm): p('   %-7s %s' % (a, json.dumps(nm[a], sort_keys=True)))

def load_engine():
    global M
    path = os.path.join('src', 'execution', 'ns_engine.py')
    spec = importlib.util.spec_from_file_location('ns_engine_check', path)
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    p('   engine loaded from %s: PKG_AHEAD_DAYS=%s PKG_NEAR=%s AHEAD_ON_WICKS=%s REPLAY_DAYS=%s DIAG_RECENT_H=%s'
      % (path, M.PKG_AHEAD_DAYS, M.PKG_NEAR, M.AHEAD_ON_WICKS, M.REPLAY_DAYS, getattr(M, 'DIAG_RECENT_H', '?')))

_PX = {}
def prices(asset):
    import pandas as pd
    if asset in _PX: return _PX[asset]
    sym = ((CFG.get('assets') or {}).get(asset) or {}).get('mt5_symbol') or ((CFG.get('assets') or {}).get(asset) or {}).get('symbol') or asset
    fs = sorted(glob.glob(os.path.join('data', 'raw', '%s_1h.csv' % sym))) or sorted(glob.glob(os.path.join('data', 'raw', '%s*_1h.csv' % asset)))
    if not fs: _PX[asset] = (None, 'no data\\raw\\%s_1h.csv' % sym); return _PX[asset]
    df = pd.read_csv(fs[0])
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in ('timestamp', 'time', 'datetime', 'date', 'open_time') if c in df.columns), df.columns[0])
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts): idx = pd.to_datetime(ts, unit='s' if float(ts.max()) < 1e11 else 'ms', utc=True)
    else: idx = pd.to_datetime(ts, utc=True, errors='coerce')
    df.index = pd.DatetimeIndex(idx).tz_convert('UTC').tz_localize(None)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    df = df[~df.index.isna()].sort_index(); df = df[~df.index.duplicated(keep='last')].dropna()
    _PX[asset] = (df, os.path.basename(fs[0])); return _PX[asset]
def frame(df, t, days):
    import pandas as pd
    t = pd.Timestamp(t); now = t + pd.Timedelta(minutes=5)
    start = (now - pd.Timedelta(days=days)).normalize()          # the bot asks MT5 from midnight UTC, `days` back
    return df[(df.index >= start) & (df.index <= t - pd.Timedelta(hours=1))]
def ahead(asset, f, d, e, t, wicks, diag_rule):
    import pandas as pd
    df4 = M.hourly_to_4h(f); t4 = M.close_times(df4, 4)
    if len(t4) < 2 * M.K + 2: return None
    hi4, lo4, c4 = (df4[c].astype(float).values for c in ('high', 'low', 'close'))
    a4 = M.atr14(hi4, lo4, c4)
    eng = M.NSEngine(asset)
    eng._sw4 = M.swings_4h(t4, hi4, lo4, c4); eng._sw4w = M.wick_swings_4h(t4, hi4, lo4); eng._t4a4 = (t4, a4)
    eng._dbrk4 = M.diagonal_breaks_4h(t4, a4, c4)
    keep = M.AHEAD_ON_WICKS; M.AHEAD_ON_WICKS = wicks
    try: lvl, edg, dist = eng._bigger_ahead(d, e, pd.Timestamp(t))
    finally: M.AHEAD_ON_WICKS = keep
    conf = None
    if lvl is not None:
        sw = eng._sw4w if wicks else eng._sw4
        cs = [s[0] for s in sw if s[2] == lvl and s[1] == ('H' if d == 1 else 'L') and s[0] <= pd.Timestamp(t)]
        conf = max(cs) if cs else None
    diag = eng._diag_recent(d, pd.Timestamp(t))[0] if diag_rule else False
    route = 'package' if (lvl is not None and dist <= M.PKG_NEAR and not diag) else 'at once'
    return lvl, dist, conf, diag, route, f.index[0]

B13_LIVE = dt.datetime(2026, 10, 7, 8, 42)       # the B13 restart, 10:42 box = 08:42 UTC
PKG_LIVE = dt.datetime(2026, 10, 2, 21, 50)      # the package went live (the forward test's B13PKG_SINCE)
def secC4():
    p('== C4. every live proof taken at once since the package went live: what 60 days would have seen ==')
    p('   (the rule that was live at the time: before the B13 restart closes and no diagonal rule; after it wicks + diagonal rule)')
    RX = re.compile(r'\[NS-PROOF\] (\w+): (\w+) (\w+) dir=([+-]?\d) R2=([\d.]+) entry=([\d.]+) .*\((\w+), (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\)')
    seen, rows = set(), []
    for f in logs_since(dt.datetime(2026, 10, 2)):
        with open(f, encoding='utf-8', errors='replace') as fh:
            for raw in fh:
                if '[NS-PROOF]' not in raw: continue
                m = RX.search(raw)
                if not m: continue
                a, tier, kind, d, r2, e, why, t = m.groups()
                k = (a, d, r2, t)
                if k in seen or tier != 'RETEST': continue
                seen.add(k); rows.append((a, tier, kind, int(d), float(r2), float(e), why, t, raw[:19]))
    rows = [r for r in rows if dt.datetime.strptime(r[7], '%Y-%m-%d %H:%M:%S') >= PKG_LIVE]
    p('   %d live RETEST proof(s) found' % len(rows))
    for a, tier, kind, d, r2, e, why, t, logged in rows:
        df, src = prices(a)
        if df is None: p('   %s %s dir=%+d: %s' % (t, a, d, src)); continue
        tt = dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        b13 = tt >= B13_LIVE
        wicks, diag = (bool(M.AHEAD_ON_WICKS), True) if b13 else (False, False)
        out = []
        for days in (25, 75):
            r = ahead(a, frame(df, t, days), d, e, t, wicks, diag)
            if r is None: out.append('%dd: too few candles' % days); continue
            lvl, dist, conf, dg, route, f0 = r
            out.append('%dd(from %s): %s -> %s' % (days, str(f0)[:10], ('no old line ahead' if lvl is None else 'old %s %.6g at %.2f moves, confirmed %s (%.0f days before)' % (
                'high' if d == 1 else 'low', lvl, dist, str(conf)[:16], (tt - conf.to_pydatetime()).total_seconds() / 86400 if conf is not None else float('nan'))) + (' +diagonal' if dg else ''), route))
        verdict = 'CHANGED by the 25-day window' if ('-> package' in out[1] and '-> at once' in out[0]) else ('CHECK: the 25-day rebuild says package but live took it at once' if '-> package' in out[0] else 'same')
        p('   %s UTC  %-6s %s dir=%+d R2=%-9.6g entry=%-9.6g [%s, %s]  => %s' % (t, a, kind, d, r2, e, 'B13' if b13 else 'before B13', src, verdict))
        for o in out: p('        ' + o)

SPOTS = [('USOIL', -1, 88.657, 0.0006, '2026-10-06 07:00:00', 'USOIL sell -- live took it at once and lost -$10.88'),
         ('USOIL', 1, 88.869, 0.0006, '2026-10-07 02:00:00', 'USOIL buy -- live made the proof; it was not traded'),
         ('EURJPY', 1, 177.664, 0.0006, '2026-10-06 12:00:00', 'EURJPY buy -- the rules took it; MT5 did not'),
         ('GOLD', -1, 4256.82, 0.006, '2026-10-07 13:00:00', 'GOLD sell -- the rules sent it to the package (entered 15:30); MT5 did not'),
         ('BTC', -1, 84741.0, 1.0, '2026-10-07 10:00:00', 'BTC sell -- live handed it to the package (cancelled); the replays never make it')]
def secC5():
    import pandas as pd
    p('== C5. spot checks: today\'s engine started fresh on 25 days (as live) and on 75 days, at each signal time ==')
    cap = []
    class _H(logging.Handler):
        def emit(self, r):
            try: cap.append(r.getMessage())
            except Exception: pass
    lg = logging.getLogger('ns_engine_check'); lg.addHandler(_H()); lg.setLevel(logging.INFO); lg.propagate = False
    for a, d, r2, tol, t, what in SPOTS:
        p('-- %s, signal %s UTC (line %g) --' % (what, t, r2))
        df, src = prices(a)
        if df is None: p('   ' + src); continue
        if df.index[-1] < pd.Timestamp(t) - pd.Timedelta(hours=1): p('   %s ends %s, before the signal' % (src, df.index[-1])); continue
        for days in (25, 75):
            f = frame(df, t, days)
            eng = M.NSEngine(a); deaths = []
            _o = eng._end
            def _rec(st, s, reason, emit, tt, _o=_o, _d=deaths): _d.append((str(tt)[:16], s.get('d'), float(s.get('r2', 0)), reason)); return _o(st, s, reason, emit, tt)
            eng._end = _rec
            cap.clear()
            try: res = eng.update(None, f, None, PC, None, now=pd.Timestamp(t) + pd.Timedelta(minutes=5))
            except Exception as ex: p('   %dd: engine stopped: %r' % (days, ex)); continue
            st = res.get('state') or {}
            near = lambda x: abs(float(x) - r2) <= tol
            live = [s for s in st.get('setups', []) if s.get('d') == d and near(s.get('r2', 0))]
            seen = [x for x in st.get('seen', []) if x[0] == d and near(x[1])]
            died = [x for x in deaths if x[1] == d and near(x[2])]
            pk = [x for x in st.get('pkg', []) if x.get('d') == d and near(x.get('r2', 0)) and str(x.get('tE'))[:16] == str(pd.Timestamp(t))[:16]]
            pr = [x for x in res.get('proofs', []) if x.get('dir') == d and near(x.get('ref', 0))]
            p('   %dd (%s -> %s, %d candles): setup on the line: %s | ended: %s | at the signal: %s' % (
                days, str(f.index[0])[:10], str(f.index[-1])[:16], len(f),
                ('alive, stage %s' % live[0].get('stage')) if live else ('made earlier, not alive' if seen else 'never made'),
                ('; '.join('%s %s' % (x[0], x[3]) for x in died[-3:]) or 'no'),
                ('PROOF (taken at once)' if pr else ('handed to the package (%s)' % pk[0].get('route') if pk else 'nothing')) ))
            for m in cap:
                if any(x in m for x in ('[NS-PROOF]', '[PKG-', '[DIAG-CONFIRM]', '[NS-SKIP]', '[KILL-R1]', '[COUNT-3-CHECK]', '[NS] ', '[NS-BRAIN]')): p('        ' + cut(m, 230))

for name, fx in [('C1', secC1), ('C2', secC2), ('C3', secC3), ('engine', load_engine), ('C4', secC4), ('C5', secC5)]:
    try:
        fx(); p()
    except Exception as e:
        p('   !! part %s stopped: %r' % (name, e)); p()
        if name == 'engine': break
p('END OF window_25d_check')
'@ | Set-Content -Path $tmpC -Encoding UTF8
& { & $Py $tmpC $Root } 2>&1 | Out-File -FilePath $outC -Encoding utf8 -Width 400
Remove-Item $tmpC -ErrorAction SilentlyContinue
"WRITTEN: $outC"
