<#
.SYNOPSIS
    Run crack segmentation training and save full execution log.

.DESCRIPTION
    Activates CrackSeg conda env, runs training, streams output to
    both console and timestamped log file.

.PARAMETER Config
    Path to model config YAML. Default: configs/ppliteseg.yaml

.PARAMETER All
    Train all three CrackSeg models in sequence (DeepLabV3+, PP-LiteSeg, PIDNet).

.PARAMETER LogDir
    Directory to save log files. Default: outputs/logs

.PARAMETER Env
    Conda environment name. Default: CrackSeg

.EXAMPLE
    # Single model
    .\scripts\train_with_log.ps1 -Config configs/ppliteseg.yaml

    # All models
    .\scripts\train_with_log.ps1 -All

    # Custom log dir
    .\scripts\train_with_log.ps1 -Config configs/deeplabv3plus.yaml -LogDir D:\logs
#>

param(
    [string]$Config  = "configs/ppliteseg.yaml",
    [switch]$All,
    [string]$LogDir  = "outputs/logs",
    [string]$Env     = "CrackSeg"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Section([string]$msg, [string]$color = "Cyan") {
    $line = "=" * 60
    Write-Host "`n$line"          -ForegroundColor $color
    Write-Host "  $msg"           -ForegroundColor $color
    Write-Host "$line"            -ForegroundColor $color
}

function Run-Training([string]$config, [string]$logDir, [string]$env) {
    $modelName = [System.IO.Path]::GetFileNameWithoutExtension($config)
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile   = Join-Path $logDir "${timestamp}_${modelName}.log"

    $startTime = Get-Date
    $header = @"
============================================================
  Training Log
  Model   : $modelName
  Config  : $config
  Env     : $env
  Start   : $($startTime.ToString("yyyy-MM-dd HH:mm:ss"))
  Log     : $logFile
============================================================
"@

    Write-Section "Starting: $modelName"
    Write-Host $header
    $header | Out-File -FilePath $logFile -Encoding utf8

    # Stream output to both console and log file
    conda run -n $env --no-capture-output `
        python training/train_crackseg.py --config $config 2>&1 `
        | Tee-Object -FilePath $logFile -Append

    $exitCode  = $LASTEXITCODE
    $endTime   = Get-Date
    $duration  = $endTime - $startTime
    $durationStr = "{0:hh\:mm\:ss}" -f $duration

    $footer = @"

============================================================
  End      : $($endTime.ToString("yyyy-MM-dd HH:mm:ss"))
  Duration : $durationStr
  Status   : $(if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED (exit $exitCode)" })
  Log saved: $logFile
============================================================
"@

    $footerColor = if ($exitCode -eq 0) { "Green" } else { "Red" }
    Write-Host $footer -ForegroundColor $footerColor
    $footer | Out-File -FilePath $logFile -Encoding utf8 -Append

    return $exitCode
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Resolve project root (script is in scripts/, so parent is root)
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Create log directory
$null = New-Item -ItemType Directory -Force -Path $LogDir

if ($All) {
    $configs = @(
        "configs/deeplabv3plus.yaml",
        "configs/ppliteseg.yaml",
        "configs/pidnet.yaml"
    )
    $sessionStart = Get-Date
    $results      = @()

    foreach ($cfg in $configs) {
        $code = Run-Training -config $cfg -logDir $LogDir -env $Env
        $results += [PSCustomObject]@{
            Model  = [System.IO.Path]::GetFileNameWithoutExtension($cfg)
            Status = if ($code -eq 0) { "OK" } else { "FAILED" }
            Code   = $code
        }
        if ($code -ne 0) {
            Write-Host "`n[ABORT] $cfg failed — stopping sequence." -ForegroundColor Red
            break
        }
    }

    # Session summary
    $sessionDuration = (Get-Date) - $sessionStart
    Write-Section "Session Summary" "Yellow"
    $results | Format-Table -AutoSize
    Write-Host ("Total time: {0:hh\:mm\:ss}" -f $sessionDuration) -ForegroundColor Yellow

    $failCount = ($results | Where-Object { $_.Status -ne "OK" }).Count
    exit $failCount

} else {
    $code = Run-Training -config $Config -logDir $LogDir -env $Env
    exit $code
}
