# =============================================================================================
#  TBOT CHECKS, 8 OCT  --  READ-ONLY: changes nothing, safe while the bot is running. A few minutes.
#  Writes two files for Claude:
#     C:\TradingBot\TBOT\b13_final_output.txt   B13 step 20, now reading every rotated daily log
#     C:\TradingBot\TBOT\usoil_0704_check.txt   why the 7 Oct USOIL buy (04:03 box time) was not traded,
#                                               plus EURJPY 6 Oct, GOLD 7 Oct and how old their lines were
#  HOW TO RUN (Stephen):
#     1. Save this file as  C:\TradingBot\TBOT\tools\tbot_checks_8oct.ps1
#     2. Open PowerShell AS ADMINISTRATOR (so the task list and process lines are complete), then:
#           cd C:\TradingBot\TBOT
#           powershell -ExecutionPolicy Bypass -File tools\tbot_checks_8oct.ps1
#     3. It ends with two lines starting "WRITTEN:". Send both .txt files to Desire.
#  Do not start it between 23:50 and 00:10 (the bot's log changes to a new file at midnight).
#  STOP and send a screenshot if there is no "WRITTEN:" line at the end.
# =============================================================================================
param([string]$Root = "C:\TradingBot\TBOT", [string]$Py = "")
$ErrorActionPreference = "Continue"
Set-Location $Root
if (-not $Py) { $Py = Join-Path $Root "venv\Scripts\python.exe" }
$outA = Join-Path $Root "b13_final_output.txt"
$outB = Join-Path $Root "usoil_0704_check.txt"
"Working... (part A of 2)"

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
"-- where ny_open / the old midnight summary live --"; Get-ChildItem -Path $Root -Recurse -Include *.py,*.ps1,*.bat,*.json -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike "*venv*" -and $_.FullName -notlike "*backup*" -and $_.FullName -ne $PSCommandPath } | Select-String -Pattern "ny_open|alarm summary|ALARM SUMMARY" -List | Select-Object Path, LineNumber, Line
"== 7. notes from steps 11 and 16 (type them here if any) =="
} 2>&1 | Out-File -FilePath $outA -Encoding utf8 -Width 400
"WRITTEN: $outA"
"Working... (part B of 2)"

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
C = dict(hours={}, cb=[], cbr={}, misc=[], w4=[], w5=[], ew=[], ewc={}, eur=[], gold=[], g48=[], gwin=[])

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
        if '2026-10-06 13:55:00' <= t < '2026-10-06 14:16:00' and not noise(s) and (key(n, l) or 'EURJPY' in s) and not ('[TRADE ASSET] Processing' in s and 'EURJPY' not in s): C['eur'].append((fn, raw))
        if t[:10] == '2026-10-07' and n is not None and 'GOLD' in s and any(x in s for x in GTAGS): C['gold'].append((fn, raw))
        if '4256.8' in s: C['g48'].append((fn, raw))
        if ('2026-10-07 14:55:00' <= t < '2026-10-07 15:16:00' or '2026-10-07 17:25:00' <= t < '2026-10-07 17:46:00') and not noise(s) and (key(n, l) or 'GOLD' in s) and not ('[TRADE ASSET] Processing' in s and 'GOLD' not in s): C['gwin'].append((fn, raw))
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
    show(C['gwin'], 200, '-- 12c. GOLD 7 Oct 14:55-15:15 and 17:25-17:45 box: GOLD + main.py + warning lines --')
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
Get-ChildItem -Path "$Root\*.ps1", "$Root\*.bat", "$Root\scripts\*.ps1", "$Root\scripts\*.bat" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -ne $PSCommandPath } | Select-String -Pattern "TBOT_|python|main\.py|Tee-Object|config\.json" | ForEach-Object { "   {0}:{1}: {2}" -f $_.Filename, $_.LineNumber, $_.Line.Trim() }
""
& $Py $tmp $Root
} 2>&1 | Out-File -FilePath $outB -Encoding utf8 -Width 400
Remove-Item $tmp -ErrorAction SilentlyContinue
"WRITTEN: $outB"
