# Separate read-only training monitor; only its readable log is written.
[CmdletBinding()]
param(
    [string]$StudyRoot,
    [string]$ManifestPath,
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 2,
    [ValidateRange(1, 100)][int]$HistoryRows = 5,
    [ValidateRange(0, 1000000)][int]$PollCount = 0,
    [switch]$Once,
    [switch]$NoFileLog
)

$ErrorActionPreference = 'Stop'
# Resolve defaults after binding: PSScriptRoot can be empty in param defaults
# when Windows PowerShell starts this script through powershell.exe -File.
$projectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($projectRoot)) {
    $scriptPath = $PSCommandPath
    if ([string]::IsNullOrWhiteSpace($scriptPath)) { $scriptPath = $MyInvocation.MyCommand.Path }
    if ([string]::IsNullOrWhiteSpace($scriptPath)) { throw 'Cannot determine viewer script directory.' }
    $projectRoot = Split-Path -Parent $scriptPath
}
if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $projectRoot 'outputs\study_two_models'
}
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $projectRoot 'configs\study\manifest.json'
}
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$readable = Join-Path $StudyRoot 'logs\training_readable.log'
$epochSeen = @{}
$errorSeen = @{}
$errorLast = @{}
$runStates = @{}
$lastStatus = ''
$lastWarning = ''
$lastMetricRun = ''

function Read-Json($Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { return $null } # Atomic replacement or a writer in progress: retry next poll.
}

function Emit([string]$Text, [string]$Kind = 'INFO') {
    $line = '[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Kind, $Text
    $color = switch ($Kind) {
        'ERROR' { 'Red' }; 'WARN' { 'Yellow' }; 'EVENT' { 'Cyan' }
        'HISTORY' { 'DarkGray' }; default { 'Gray' }
    }
    Write-Host $line -ForegroundColor $color
    if (-not $NoFileLog) {
        try { Add-Content -LiteralPath $readable -Value $line -Encoding UTF8 }
        catch { Write-Warning "Cannot append viewer log: $($_.Exception.Message)" }
    }
}

function Number($Value, [string]$Format = 'F2') {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return '--' }
    $n = 0.0
    if ([double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $culture, [ref]$n)) {
        return $n.ToString($Format, $culture)
    }
    return '--'
}

function Process-State($ProcessId) {
    if (-not $ProcessId) { return '--' }
    $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($p) { return "$ProcessId/$($p.ProcessName) alive" }
    return "$ProcessId not running"
}

function Show-Overview($Runs) {
    Emit 'GRID OVERVIEW (validation only; smoke excluded)' 'EVENT'
    Emit ('{0,-44} {1,-12} {2,9} {3,11} {4,-8}' -f 'Run', 'State', 'Epochs', 'Best val %', 'Export')
    foreach ($run in $Runs) {
        $count = 0
        if ($run.Summary) { $count = [int]$run.Summary.completed_epochs }
        $state = 'pending'
        if ($count -gt 0) { $state = 'partial' }
        if ($count -eq $run.Config.epochs) { $state = 'trained' }
        if ($run.Exported) { $state = 'done' }
        $export = if ($run.Exported) { 'complete' } else { '--' }
        $best = if ($run.Summary) { Number $run.Summary.best_validation_accuracy } else { '--' }
        Emit ('{0,-44} {1,-12} {2,9} {3,11} {4,-8}' -f $run.Name, $state, "$count/$($run.Config.epochs)", $best, $export)
    }
}

if (-not $NoFileLog) {
    [void](New-Item -ItemType Directory -Force -Path (Split-Path $readable))
}
$manifest = Read-Json $ManifestPath
if (-not $manifest -or -not $manifest.grid) { throw "Cannot read grid manifest: $ManifestPath" }
$configs = @()
foreach ($path in $manifest.grid) {
    $config = Read-Json $path
    if (-not $config) { throw "Cannot read config: $path" }
    $output = $config.output_dir
    if (-not [System.IO.Path]::IsPathRooted($output)) { $output = Join-Path $projectRoot $output }
    $configs += [PSCustomObject]@{ Config = $config; Output = $output; Name = (Split-Path $output -Leaf) }
}
Emit 'VIEWER SESSION START. Timestamps = observation time, not original epoch time.' 'EVENT'
Emit 'History is labeled; new epochs appear once per session. Ctrl+C stops ONLY this viewer.'
Emit 'Metrics update after saved epochs, not batches. Train+Val excludes checkpoint overhead.'
if (-not $NoFileLog) { Emit "Readable log: $readable" }

$poll = 0
do {
    $status = Read-Json (Join-Path $StudyRoot 'status.json')
    $runs = @()
    $trained = 0
    $exported = 0
    foreach ($entry in $configs) {
        $summary = Read-Json (Join-Path $entry.Output 'summary.json')
        $complete = Test-Path (Join-Path $entry.Output 'evaluation_validation\complete.json')
        if ($summary -and $summary.completed_epochs -eq $entry.Config.epochs) { $trained++ }
        if ($complete) { $exported++ }
        $runs += [PSCustomObject]@{ Name = $entry.Name; Config = $entry.Config; Output = $entry.Output; Summary = $summary; Exported = $complete }
    }
    if (-not $status) {
        if ($lastWarning -ne 'status') { Emit 'Status temporarily unavailable; retrying.' 'WARN'; $lastWarning = 'status' }
    } else {
        $lastWarning = ''
        $queueState = Process-State $status.pid
        $childState = Process-State $status.child_pid
        $key = "$($status.phase)|$($status.current_run)|$trained|$exported|$queueState|$childState"
        if ($key -ne $lastStatus) {
            Emit "QUEUE: $($status.phase) | Trained $trained/$($configs.Count) | Exports $exported/$($configs.Count)" 'EVENT'
            Emit "RUN: $($status.current_run) | Queue $queueState | Child $childState"
            if ($queueState -like '*not running' -and $status.phase -in @('training', 'exporting_validation', 'starting')) {
                Emit 'Status says active but queue PID is absent; inspect stderr/checkpoint before resuming.' 'WARN'
            }
            if ($status.error) { Emit ([string]$status.error) 'ERROR' }
            Show-Overview $runs
            $lastStatus = $key
        }
        foreach ($run in $runs) {
            $state = "$($run.Summary.completed_epochs)|$($run.Exported)"
            if ($runStates.ContainsKey($run.Name) -and $runStates[$run.Name] -ne $state -and $run.Exported) {
                Emit "RUN COMPLETE + VALIDATION EXPORT: $($run.Name)" 'EVENT'
            }
            $runStates[$run.Name] = $state
            # Track previous active runs too, so their final epoch is not lost at a switch.
            if ($run.Name -ne $status.current_run -and -not $epochSeen.ContainsKey($run.Name)) { continue }
            if (-not $run.Summary) { continue }
            $csv = Join-Path $run.Output 'metrics.csv'
            if (-not (Test-Path $csv)) { continue }
            try {
                $rows = @(Import-Csv -LiteralPath $csv -Encoding UTF8 | Where-Object {
                    $_.epoch -match '^\d+$' -and [int]$_.epoch -le [int]$run.Summary.completed_epochs
                })
            } catch { continue }
            $isHistory = -not $epochSeen.ContainsKey($run.Name)
            if ($isHistory) {
                $epochSeen[$run.Name] = 0
                $rows = @($rows | Select-Object -Last $HistoryRows)
            }
            foreach ($m in $rows) {
                if ([int]$m.epoch -le $epochSeen[$run.Name]) { continue }
                if ($lastMetricRun -ne $run.Name) {
                    Emit "METRICS: $($run.Name) | $($run.Config.dataset) | $($run.Config.variant) | $($run.Config.optimizer) | Initial LR $(Number $run.Config.lr 'G6')" 'EVENT'
                    Emit 'Epoch    TrainLoss TrainAcc%   ValLoss  ValAcc%         LR  Train+Val(s) AMPskip GPUalloc/res(MiB)'
                    $lastMetricRun = $run.Name
                }
                $seconds = [double]::Parse($m.train_seconds, $culture) + [double]::Parse($m.validation_seconds, $culture)
                $allocated = [double]::Parse($m.peak_allocated_bytes, $culture) / 1MB
                $reserved = [double]::Parse($m.peak_reserved_bytes, $culture) / 1MB
                $kind = if ($isHistory) { 'HISTORY' } else { 'EPOCH' }
                Emit ('{0,-9} {1,9} {2,9} {3,9} {4,8} {5,10} {6,13} {7,7} {8,7}/{9}' -f `
                    "$($m.epoch)/$($run.Config.epochs)", (Number $m.train_loss 'F4'), (Number $m.train_weighted_accuracy), `
                    (Number $m.validation_loss 'F4'), (Number $m.validation_accuracy), (Number $m.lr_last 'G6'), `
                    (Number $seconds 'F1'), $m.skipped_amp_steps, (Number $allocated 'F0'), (Number $reserved 'F0')) $kind
                $epochSeen[$run.Name] = [int]$m.epoch
            }
        }
    }
    # Follow grid stderr and development queue logs, not old smoke failures.
    $errorFiles = @()
    foreach ($entry in $configs) {
        $path = Join-Path $StudyRoot "logs\$($entry.Name).stderr.log"
        if (Test-Path $path) { $errorFiles += Get-Item $path }
    }
    $errorFiles += @(Get-ChildItem -Path (Join-Path $StudyRoot 'queue*.stderr.log') -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -notlike '*smoke*'
    })
    foreach ($file in $errorFiles) {
        try {
            $raw = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
            $lines = @()
            if ($raw) {
                $parts = $raw -split "`n"
                # Leave an incomplete last line for the next poll.
                if ($parts.Count -gt 1) { $lines = @($parts[0..($parts.Count - 2)]) }
            }
            $historical = -not $errorSeen.ContainsKey($file.FullName)
            if ($historical) { $errorSeen[$file.FullName] = [Math]::Max(0, $lines.Count - 10) }
            if ($lines.Count -lt $errorSeen[$file.FullName]) { $errorSeen[$file.FullName] = 0 }
            $offset = $errorSeen[$file.FullName]
            if ($offset -gt 0 -and $errorLast.ContainsKey($file.FullName) -and $lines[$offset - 1] -ne $errorLast[$file.FullName]) {
                $errorSeen[$file.FullName] = 0 # File truncated/replaced between polls.
            }
            for ($i = $errorSeen[$file.FullName]; $i -lt $lines.Count; $i++) {
                if ([string]::IsNullOrWhiteSpace($lines[$i])) { continue }
                $prefix = if ($historical) { 'HISTORY STDERR' } else { 'STDERR' }
                Emit "$prefix $($file.Name): $($lines[$i].TrimEnd())" 'ERROR'
            }
            $errorSeen[$file.FullName] = $lines.Count
            if ($lines.Count -gt 0) { $errorLast[$file.FullName] = $lines[-1] }
        } catch { } # Log may be replaced while reading; retry.
    }
    $poll++
    if ($Once -or ($PollCount -gt 0 -and $poll -ge $PollCount)) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)