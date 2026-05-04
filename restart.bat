@echo off
echo [TBOT] Stopping scheduled task...
powershell -Command "Stop-ScheduledTask -TaskName 'TBOT' -ErrorAction SilentlyContinue"

echo [TBOT] Killing all Python processes...
taskkill /F /IM python.exe 2>nul
if %errorlevel%==0 (
    echo [TBOT] Python processes killed.
) else (
    echo [TBOT] No Python processes found.
)

timeout /t 2 /nobreak >nul

echo [TBOT] Starting bot via Task Scheduler...
powershell -Command "Start-ScheduledTask -TaskName 'TBOT'"
echo [TBOT] Done. Bot starting in background via Task Scheduler.
