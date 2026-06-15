# Trading Bot Wrapper Script
# This script runs your bot and handles logging

$LogFile = "C:\Users\USER\TBOT\logs\bot_$(Get-Date -Format 'yyyyMMdd').log"
$ErrorLog = "C:\Users\USER\TBOT\logs\bot_error_$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$Timestamp - $Message" | Tee-Object -FilePath $LogFile -Append
}

Write-Log "=========================================="
Write-Log "Starting Trading Bot"
Write-Log "=========================================="

try {
    Set-Location "C:\Users\USER\TBOT"

    # Wait for MT5 to initialise and connect to broker feed
    Write-Log "Waiting 60s for MetaTrader 5 to initialise..."
    Start-Sleep -Seconds 60

    Write-Log "Executing: C:\Users\USER\TBOT\venv\Scripts\python.exe C:\Users\USER\TBOT\main.py"
    & "C:\Users\USER\TBOT\venv\Scripts\python.exe" "C:\Users\USER\TBOT\main.py" 2>&1 | Tee-Object -FilePath $LogFile -Append

    $ExitCode = $LASTEXITCODE
    Write-Log "Bot exited with code: $ExitCode"

    if ($ExitCode -ne 0) {
        Write-Log "ERROR: Bot exited with non-zero code" | Tee-Object -FilePath $ErrorLog -Append
    }

} catch {
    Write-Log "EXCEPTION: $($_.Exception.Message)" | Tee-Object -FilePath $ErrorLog -Append
    Write-Log "Stack Trace: $($_.ScriptStackTrace)" | Tee-Object -FilePath $ErrorLog -Append
}

Write-Log "=========================================="
Write-Log "Bot execution completed"
Write-Log "=========================================="
