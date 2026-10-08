# Standalone PowerShell5.1 tests. Fixtures never touch the live queue or GPU.
$ErrorActionPreference = 'Stop'
$viewer = Join-Path (Split-Path $PSScriptRoot -Parent) 'watch_training.ps1'
$temp = Join-Path ([System.IO.Path]::GetTempPath()) ('lightnet_monitor_' + [guid]::NewGuid())
$job = $null

function Assert($Condition, $Message) {
    if (-not $Condition) { throw "Assertion failed: $Message" }
}
function Json($Path, $Value) {
    $Value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Path -Encoding UTF8
}
function Metric($Epoch, $Validation = 70) {
    return [PSCustomObject]@{
        epoch = $Epoch; train_loss = 0.5; train_weighted_accuracy = 80
        validation_loss = $(if ($null -eq $Validation) { $null } else { 0.6 })
        validation_accuracy = $Validation; lr_last = 0.001
        train_seconds = 2; validation_seconds = 1; skipped_amp_steps = 0
        peak_allocated_bytes = 1048576; peak_reserved_bytes = 2097152
    }
}
try {
    [void](New-Item -ItemType Directory "$temp\logs", "$temp\a", "$temp\b" -Force)
    $a = @{ output_dir = "$temp\a"; epochs = 3; dataset = 'cifar10'; variant = 'baseline'; optimizer = 'sgd'; lr = 0.1 }
    $b = @{ output_dir = "$temp\b"; epochs = 3; dataset = 'cifar100'; variant = 'multiscale'; optimizer = 'adam'; lr = 0.001 }
    Json "$temp\a.json" $a; Json "$temp\b.json" $b
    Json "$temp\manifest.json" @{ grid = @("$temp\a.json", "$temp\b.json") }
    Json "$temp\status.json" @{ phase = 'paused_by_user'; current_run = 'a'; pid = 99999999 }
    Json "$temp\a\summary.json" @{ completed_epochs = 1; best_validation_accuracy = 70 }
    @(Metric 1; Metric 2) | Export-Csv "$temp\a\metrics.csv" -NoTypeInformation
    Set-Content "$temp\queue_smoke.stderr.log" 'old smoke failure'
    # Incomplete stderr line must not be emitted until a newline arrives.
    [System.IO.File]::WriteAllText("$temp\logs\a.stderr.log", 'partial error')
    & $viewer -StudyRoot $temp -ManifestPath "$temp\manifest.json" -Once -NoFileLog *> "$temp\snapshot.txt"
    $text = Get-Content "$temp\snapshot.txt" -Raw
    Assert ($text -match 'paused_by_user') 'pause displayed'
    Assert ($text -match '\[HISTORY\] 1/3') 'saved history displayed'
    Assert ($text -notmatch '\[HISTORY\] 2/3') 'unsaved row excluded'
    Assert ($text -notmatch 'partial error|old smoke failure') 'partial errors/smoke excluded'
    Assert (-not (Test-Path "$temp\logs\training_readable.log")) 'NoFileLog does not write'

    # Background fixture writer changes epochs, exports and the active run.
    $job = Start-Job -ArgumentList $temp -ScriptBlock {
        param($t)
        Start-Sleep -Seconds 2
        [System.IO.File]::AppendAllText("$t\logs\a.stderr.log", "`n")
        @{ completed_epochs = 2; best_validation_accuracy = 71 } | ConvertTo-Json | Set-Content "$t\a\summary.json"
        Start-Sleep -Seconds 2
        Import-Csv "$t\a\metrics.csv" | Select-Object -Last 1 | ForEach-Object { $_.epoch = '3'; $_ } | Export-Csv "$t\a\metrics.csv" -NoTypeInformation -Append
        @{ completed_epochs = 3; best_validation_accuracy = 72 } | ConvertTo-Json | Set-Content "$t\a\summary.json"
        [void](New-Item -ItemType Directory "$t\a\evaluation_validation")
        '{}' | Set-Content "$t\a\evaluation_validation\complete.json"
        @{ phase = 'training'; current_run = 'b'; pid = 99999999; child_pid = 99999998 } | ConvertTo-Json | Set-Content "$t\status.json"
        $m = Import-Csv "$t\a\metrics.csv" | Select-Object -First 1
        $m.validation_accuracy = ''; $m.validation_loss = ''
        $m | Export-Csv "$t\b\metrics.csv" -NoTypeInformation
        @{ completed_epochs = 1; best_validation_accuracy = $null } | ConvertTo-Json | Set-Content "$t\b\summary.json"
        Start-Sleep -Seconds 2
        # Truncation is detected; the new complete error is still emitted.
        'replacement error' | Set-Content "$t\logs\a.stderr.log"
        'second error' | Add-Content "$t\logs\a.stderr.log"
        @{ phase = 'failed'; current_run = 'b'; error = 'fixture failure' } | ConvertTo-Json | Set-Content "$t\status.json"
    }
    & $viewer -StudyRoot $temp -ManifestPath "$temp\manifest.json" -PollCount 12 -IntervalSeconds 1 *> "$temp\watch.txt"
    $text = Get-Content "$temp\logs\training_readable.log" -Raw
    Assert (([regex]::Matches($text, '\[EPOCH\] 2/3')).Count -eq 1) 'new epoch2 once'
    Assert (([regex]::Matches($text, '\[EPOCH\] 3/3')).Count -eq 1) 'final epoch retained across switch'
    Assert ($text -match 'RUN COMPLETE \+ VALIDATION EXPORT: a') 'export event'
    Assert ($text -match 'METRICS: b') 'automatic run switch'
    Assert ($text -match 'fixture failure') 'status error'
    Assert ($text -match 'STDERR a.stderr.log: partial error') 'completed error line'
    Assert ($text -match 'second error') 'replacement stderr followed'
    Assert ($text -match 'replacement error') 'replacement first line retained'
    Assert ($text -match 'queue PID is absent') 'stale active status warning'
    Assert ($text -match '\[HISTORY\] 1/3\s+0.5000\s+80.00\s+--\s+--') 'missing validation formatted'
    Wait-Job $job | Out-Null
    Receive-Job $job -ErrorAction Stop | Out-Null

    Remove-Item "$temp\status.json"
    & $viewer -StudyRoot $temp -ManifestPath "$temp\manifest.json" -Once -NoFileLog *> "$temp\missing.txt"
    Assert ((Get-Content "$temp\missing.txt" -Raw) -match 'Status temporarily unavailable') 'missing status retry'
    Write-Host 'PASS: snapshots, saved rows, run switch, export, dedup, null validation, stderr, missing/stale status, file logging.'
} finally {
    if ($job) { Remove-Job $job -Force -ErrorAction SilentlyContinue }
    if (Test-Path $temp) { Remove-Item $temp -Recurse -Force }
}