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
    Assert ($text -match 'BestVal%\s+Progress') 'compact progress header'
    Assert ($text -match '\[HISTORY\] 1/3\s+0.5000\s+80.00\s+0.6000\s+70.00\s+70.00\s+33.3%') 'accurate per-epoch best and percentage'
    Assert ($text -notmatch 'GPUalloc|AMPskip|DETAIL:') 'technical metrics hidden by default'
    'Traceback: old memory failure' | Set-Content "$temp\logs\b.stderr.log"
    & $viewer -StudyRoot $temp -ManifestPath "$temp\manifest.json" -PollCount 2 -IntervalSeconds 1 -NoFileLog *> "$temp\history.txt"
    $history = Get-Content "$temp\history.txt" -Raw
    Assert (([regex]::Matches($history, 'HISTORICAL STDERR: b.stderr.log')).Count -eq 1) 'history collapsed once'
    Assert ($history -notmatch 'Traceback: old memory failure|\[ERROR\]') 'no historical traceback or false error'
    Assert ((Get-Content $viewer -Raw) -notmatch "'Red'") 'no red output'

    # Regression: a fresh -File process must resolve defaults from its script,
    # not the caller's unrelated working directory. Fixture paths include spaces.
    $cliRoot = Join-Path $temp 'cli project with spaces'
    $cwd = Join-Path $temp 'unrelated working directory'
    [void](New-Item -ItemType Directory "$cliRoot\configs\study", "$cliRoot\outputs\study_two_models", $cwd -Force)
    Copy-Item $viewer "$cliRoot\watch_training.ps1"
    Copy-Item "$temp\manifest.json" "$cliRoot\configs\study\manifest.json"
    Copy-Item "$temp\status.json" "$cliRoot\outputs\study_two_models\status.json"
    Push-Location $cwd
    try {
        $cliOutput = & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$cliRoot\watch_training.ps1" -Once -NoFileLog 2>&1
        $cliExit = $LASTEXITCODE
    } finally { Pop-Location }
    $cliText = $cliOutput -join "`n"
    Assert ($cliExit -eq 0) "fresh -File defaults succeed: $cliText"
    Assert ($cliText -match 'GRID OVERVIEW' -and $cliText -match '\[HISTORY\] 1/3') 'CLI reads default manifest/status from script directory'
    Assert (-not (Test-Path "$cwd\outputs")) 'CLI does not create paths under caller directory'

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
    & $viewer -StudyRoot $temp -ManifestPath "$temp\manifest.json" -PollCount 12 -IntervalSeconds 1 -Details *> "$temp\watch.txt"
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

    # Isolated two-study chain: completed smoke does not mark long trials trained.
    $newRoot = "$temp\improvement"
    [void](New-Item -ItemType Directory "$newRoot\logs", "$newRoot\trial_plain", "$newRoot\trial_cutmix" -Force)
    $plain = @{ output_dir = "$newRoot\trial_plain"; epochs = 3; dataset = 'cifar10'; variant = 'hybrid'; optimizer = 'sgd'; lr = 0.1 }
    $cutmix = @{ output_dir = "$newRoot\trial_cutmix"; epochs = 3; dataset = 'cifar10'; variant = 'hybrid'; optimizer = 'sgd'; lr = 0.1; cutmix_alpha = 1; cutmix_probability = 0.5 }
    Json "$newRoot\plain.json" $plain; Json "$newRoot\cutmix.json" $cutmix
    Json "$newRoot\manifest.json" @{ grid = @("$newRoot\plain.json", "$newRoot\cutmix.json"); output_root = $newRoot }
    Json "$temp\status.json" @{ phase = 'grid_completed_awaiting_recipe_selection'; current_run = 'a' }
    Json "$newRoot\status.json" @{ phase = 'smoke_completed'; stage = 'smoke'; current_run = 'smoke_hybrid' }
    Json "$temp\chain.json" @{ phase = 'running_queue'; manifest = "$temp\manifest.json"; pid = $PID }
    'old supervisor traceback' | Set-Content "$temp\approved_chain.stderr.log"
    $argsView = @{ StudyRoot = $temp; ManifestPath = "$temp\manifest.json"; AdditionalManifestPaths = @("$newRoot\manifest.json"); ChainStatusPath = "$temp\chain.json"; NoFileLog = $true }
    & $viewer @argsView -Once *> "$temp\chain_smoke.txt"
    $text = Get-Content "$temp\chain_smoke.txt" -Raw
    Assert ($text -match 'SMOKE ONLY') 'smoke explicitly separate'
    Assert ($text -match 'trial_plain\s+waiting\s+0/3' -and $text -match 'trial_cutmix\s+waiting\s+0/3') 'new names waiting before outputs'
    Assert ($text -notmatch 'METRICS: smoke') 'no smoke metrics as long run'
    Assert ($text -match 'HISTORICAL STDERR: approved_chain.stderr.log' -and $text -notmatch 'old supervisor traceback') 'chain stderr followed and collapsed'
    # Default discovery also works in a fresh CLI from an unrelated directory.
    [void](New-Item -ItemType Directory "$cliRoot\configs\improvement" -Force)
    Copy-Item "$newRoot\manifest.json" "$cliRoot\configs\improvement\manifest.json"
    Push-Location $cwd
    try {
        $cliText = (& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$cliRoot\watch_training.ps1" -Once -NoFileLog) -join "`n"
        Assert ($LASTEXITCODE -eq 0) 'CLI automatic improvement discovery succeeds'
    } finally { Pop-Location }
    Assert ($cliText -match 'trial_plain\s+waiting') 'default invocation discovers new experiments'

    Remove-Job $job -Force; $job = $null
    # Same live viewer follows the chain into improvement, captures final epoch,
    # then displays an actual failure without dumping the traceback by default.
    $job = Start-Job -ArgumentList $temp, $newRoot, $PID -ScriptBlock {
        param($t, $n, $livePid)
        Start-Sleep -Seconds 2
        @{ phase = 'running_queue'; manifest = "$n\manifest.json"; pid = $livePid } | ConvertTo-Json | Set-Content "$t\chain.json"
        @{ phase = 'training'; current_run = 'trial_plain'; pid = $livePid; child_pid = $livePid } | ConvertTo-Json | Set-Content "$n\status.json"
        $m = Import-Csv "$t\a\metrics.csv" | Select-Object -First 1
        $m | Export-Csv "$n\trial_plain\metrics.csv" -NoTypeInformation
        @{ completed_epochs = 1; best_validation_accuracy = 70 } | ConvertTo-Json | Set-Content "$n\trial_plain\summary.json"
        Start-Sleep -Seconds 2
        $m.epoch = '3'; $m.validation_accuracy = '72'
        $m | Export-Csv "$n\trial_plain\metrics.csv" -NoTypeInformation -Append
        @{ completed_epochs = 3; best_validation_accuracy = 72 } | ConvertTo-Json | Set-Content "$n\trial_plain\summary.json"
        [void](New-Item -ItemType Directory "$n\trial_plain\evaluation_validation")
        '{}' | Set-Content "$n\trial_plain\evaluation_validation\complete.json"
        @{ phase = 'training'; current_run = 'trial_cutmix'; pid = $livePid } | ConvertTo-Json | Set-Content "$n\status.json"
        $m.epoch = '1'; $m | Export-Csv "$n\trial_cutmix\metrics.csv" -NoTypeInformation
        @{ completed_epochs = 1; best_validation_accuracy = 72 } | ConvertTo-Json | Set-Content "$n\trial_cutmix\summary.json"
        Start-Sleep -Seconds 2
        "Traceback: new failure`nRuntimeError: fixture" | Set-Content "$n\logs\trial_cutmix.stderr.log"
        @{ phase = 'failed'; current_run = 'trial_cutmix'; error = 'confirmed fixture failure' } | ConvertTo-Json | Set-Content "$n\status.json"
        @{ phase = 'failed_or_blocked'; error = 'chain fixture failure' } | ConvertTo-Json | Set-Content "$t\chain.json"
        'new supervisor error' | Add-Content "$t\approved_chain.stderr.log"
    }
    & $viewer @argsView -PollCount 12 -IntervalSeconds 1 *> "$temp\chain_watch.txt"
    $text = Get-Content "$temp\chain_watch.txt" -Raw
    Assert ($text -match 'trial_plain\s+running') 'new trial becomes running'
    Assert ($text -match 'trial_plain\s+completed') 'new trial completion visible'
    Assert ($text -match 'RUN COMPLETE \+ VALIDATION EXPORT: trial_plain') 'new export event'
    Assert ($text -match '\[EPOCH\] 3/3') 'last saved epoch captured during transition'
    Assert ($text -match 'METRICS: trial_cutmix.*CutMix') 'new trial recipe visible'
    Assert ($text -match 'trial_cutmix\s+failed' -and $text -match 'confirmed fixture failure') 'confirmed failure distinct from warning'
    Assert ($text -match 'CHAIN STOPPED: chain fixture failure') 'chain blocked/failure visible'
    Assert ($text -match 'NEW STDERR: approved_chain.stderr.log') 'new supervisor stderr followed'
    Assert ($text -match 'NEW STDERR: trial_cutmix.stderr.log') 'newly created stderr not classified as history'
    Assert (([regex]::Matches($text, 'NEW STDERR: trial_cutmix.stderr.log')).Count -eq 1) 'new stderr deduplicated'
    Assert ($text -notmatch 'HISTORICAL STDERR: trial_cutmix.stderr.log') 'fresh traceback not hidden as old history'
    Assert ($text -notmatch 'Traceback: new failure|RuntimeError: fixture') 'default failure output collapsed'
    Wait-Job $job | Out-Null; Receive-Job $job -ErrorAction Stop | Out-Null
    Write-Host 'PASS: CLI paths/discovery, compact metrics, named waiting/running/completed/failed trials, smoke separation, chain transitions, exports, dedup, null validation, historical/new stderr, missing/stale status, file logging.'
} finally {
    if ($job) { Remove-Job $job -Force -ErrorAction SilentlyContinue }
    if (Test-Path $temp) { Remove-Item $temp -Recurse -Force }
}