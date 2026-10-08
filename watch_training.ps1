# Separate read-only training monitor; only its readable log is written.
[CmdletBinding()]
param(
    [string]$StudyRoot,
    [string]$ManifestPath,
    [string[]]$AdditionalManifestPaths = @(),
    [string]$ChainStatusPath,
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 2,
    [ValidateRange(1, 100)][int]$HistoryRows = 5,
    [ValidateRange(0, 1000000)][int]$PollCount = 0,
    [switch]$Once,
    [switch]$NoFileLog,
    [switch]$Details
)

$ErrorActionPreference = 'Stop'
$autoDiscover = -not $StudyRoot -and -not $ManifestPath
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
$newEpochCounts = @{}
$errorSeen = @{}
$errorLast = @{}
$runStates = @{}
$lastMetricRun = ''
$lastChain = ''
$lastStatuses = @{}
$warnings = @{}
$sessionStarted = [DateTime]::UtcNow

function Read-Json($Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { return $null } # Atomic replacement or a writer in progress: retry next poll.
}

function Emit([string]$Text, [string]$Kind = 'INFO') {
    # Keep status exceptions on one readable line; raw tracebacks stay in stderr.
    $Text = $Text -replace '[\r\n]+', ' '
    if ($Kind -eq 'ERROR' -and -not $Details -and $Text.Length -gt 240) {
        $Text = $Text.Substring(0, 240) + ' ... (see raw stderr / -Details)'
    }
    $line = '[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Kind, $Text
    $color = switch ($Kind) {
        'ERROR' { 'Yellow' }; 'WARN' { 'Yellow' }; 'EVENT' { 'Cyan' }
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

function Run-State($Run, $Status) {
    if ($Run.Exported) { return 'completed' }
    $active = $Status -and $Run.Name -eq $Status.current_run
    if ($active -and $Status.phase -match 'failed|blocked') { return 'failed' }
    if ($active -and $Status.phase -in @('training', 'exporting_validation', 'starting')) {
        if ((Process-State $Status.pid) -like '*not running') { return 'stale-status' }
        if ($Status.phase -eq 'exporting_validation') { return 'exporting' }
        return 'running'
    }
    if ($Run.Summary -and $Run.Summary.completed_epochs -ge $Run.Config.epochs) { return 'trained' }
    if ($Run.Summary -and $Run.Summary.completed_epochs -gt 0) { return 'partial' }
    return 'waiting'
}

function Recipe($Config) {
    if ($Config.cutmix_alpha -gt 0 -and $Config.cutmix_probability -gt 0) { return 'CutMix' }
    return 'Plain'
}

function Show-Overview($Runs, $Status, $Label) {
    Emit "STUDY: $Label" 'EVENT'
    Emit 'GRID OVERVIEW (validation only; smoke excluded)' 'EVENT'
    Emit ('{0,-54} {1,-12} {2,9} {3,11} {4,-8}' -f 'Experiment / run ID', 'State', 'Epochs', 'Best val %', 'Export')
    foreach ($run in $Runs) {
        $count = 0
        if ($run.Summary) { $count = [int]$run.Summary.completed_epochs }
        $state = Run-State $run $Status
        $export = if ($run.Exported) { 'complete' } else { '--' }
        $best = if ($run.Summary) { Number $run.Summary.best_validation_accuracy } else { '--' }
        Emit ('{0,-54} {1,-12} {2,9} {3,11} {4,-8}' -f $run.Name, $state, "$count/$($run.Config.epochs)", $best, $export)
    }
}

function Show-MetricHeader($Run) {
    Emit "METRICS: $($Run.Name) | $($Run.Config.dataset) | $($Run.Config.variant) | $(Recipe $Run.Config) | $($Run.Config.optimizer) | Initial LR $(Number $Run.Config.lr 'G6')" 'EVENT'
    Emit "SAVED PROGRESS: $($Run.Summary.completed_epochs)/$($Run.Config.epochs) | Best validation $(Number $Run.Summary.best_validation_accuracy)%"
    Emit 'Epoch    TrainLoss TrainAcc%   ValLoss  ValAcc%  BestVal%   Progress Train+Val(s)'
}

if (-not $NoFileLog) {
    [void](New-Item -ItemType Directory -Force -Path (Split-Path $readable))
}
$manifestPaths = @($ManifestPath) + $AdditionalManifestPaths
if ($autoDiscover) {
    $improvement = Join-Path $projectRoot 'configs\improvement\manifest.json'
    if (Test-Path -LiteralPath $improvement) { $manifestPaths += $improvement }
    $redesign = Join-Path $projectRoot 'configs\redesign\manifest.json'
    if (Test-Path -LiteralPath $redesign) { $manifestPaths += $redesign }
    if (-not $ChainStatusPath) { $ChainStatusPath = Join-Path $projectRoot 'outputs\approved_chain_status.json' }
}
$studies = @()
foreach ($manifestFile in ($manifestPaths | Select-Object -Unique)) {
    $manifest = Read-Json $manifestFile
    if (-not $manifest -or -not $manifest.grid) { throw "Cannot read grid manifest: $manifestFile" }
    $root = $StudyRoot
    if ($manifestFile -ne $ManifestPath) {
        $root = $manifest.output_root
        if (-not $root) { throw "Missing output_root: $manifestFile" }
        if (-not [System.IO.Path]::IsPathRooted($root)) { $root = Join-Path $projectRoot $root }
    }
    $configs = @()
    foreach ($path in $manifest.grid) {
        if (-not [System.IO.Path]::IsPathRooted($path)) { $path = Join-Path $projectRoot $path }
        $config = Read-Json $path
        if (-not $config) { throw "Cannot read config: $path" }
        $output = $config.output_dir
        if (-not [System.IO.Path]::IsPathRooted($output)) { $output = Join-Path $projectRoot $output }
        $configs += [PSCustomObject]@{ Config = $config; Output = $output; Name = (Split-Path $output -Leaf) }
    }
    $studies += [PSCustomObject]@{ Root = $root; Configs = $configs; Label = (Split-Path $root -Leaf) }
}
Emit 'VIEWER SESSION START. Timestamps = observation time, not original epoch time.' 'EVENT'
Emit 'History is labeled; new epochs appear once per session. Ctrl+C stops ONLY this viewer.'
Emit 'Metrics update after saved epochs, not batches. Train+Val excludes checkpoint overhead.'
Emit 'Smoke completion is NOT long-run completion. Historical stderr is collapsed; new stderr is a warning, not proof of failure.'
Emit 'Default view: compact metrics, neutral colors. -Details includes LR/AMP/VRAM and stderr text.'
Emit 'Experiment overview and model header repeat every 5 new saved epochs of the current training run.'
if (-not $NoFileLog) { Emit "Readable log: $readable" }

$poll = 0
do {
    if ($ChainStatusPath) {
        $chain = Read-Json $ChainStatusPath
        if ($chain) {
            $chainKey = "$($chain.phase)|$($chain.manifest)|$($chain.error)|$(Process-State $chain.pid)"
            if ($chainKey -ne $lastChain) {
                Emit "CHAIN: $($chain.phase) | Supervisor $(Process-State $chain.pid)" 'EVENT'
                Emit "CHAIN STAGE: $($chain.manifest)"
                if ($chain.error) { Emit "CHAIN STOPPED: $($chain.error)" 'ERROR' }
                $lastChain = $chainKey
            }
        } elseif (-not $warnings.ContainsKey('chain')) {
            Emit 'Chain status temporarily unavailable; study queues are shown independently.' 'WARN'
            $warnings['chain'] = $true
        }
    }
    foreach ($study in $studies) {
    $configs = $study.Configs
    $status = Read-Json (Join-Path $study.Root 'status.json')
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
        if (-not $warnings.ContainsKey($study.Root)) { Emit "Status temporarily unavailable; retrying. Study: $($study.Label)" 'WARN'; $warnings[$study.Root] = $true }
        if (-not $lastStatuses.ContainsKey($study.Root)) {
            Show-Overview $runs $null $study.Label
            $lastStatuses[$study.Root] = 'missing'
        }
    } else {
        $warnings.Remove($study.Root)
        $queueState = Process-State $status.pid
        $childState = Process-State $status.child_pid
        $key = "$($status.phase)|$($status.current_run)|$trained|$exported|$queueState|$childState|$($status.error)"
        if ($key -ne $lastStatuses[$study.Root]) {
            Emit "QUEUE: $($status.phase) | Study $($study.Label) | Retained-grid trained $trained/$($configs.Count) | Exports $exported/$($configs.Count)" 'EVENT'
            Emit "RUN: $($status.current_run) | Queue $queueState | Child $childState"
            $current = $runs | Where-Object { $_.Name -eq $status.current_run } | Select-Object -First 1
            if ($current) {
                $count = 0
                if ($current.Summary) { $count = [int]$current.Summary.completed_epochs }
                Emit "CURRENT EXPERIMENT: $($current.Config.dataset) | $($current.Config.variant) | $(Recipe $current.Config) | $($current.Config.optimizer) LR $(Number $current.Config.lr 'G6') | $(Run-State $current $status)"
                Emit "PROGRESS: $count/$($current.Config.epochs) saved epochs | Best validation $(Number $current.Summary.best_validation_accuracy)%"
            }
            if ($status.stage -eq 'smoke' -or $status.phase -eq 'smoke_completed') {
                Emit 'SMOKE ONLY: preflight status above; long trials below are separate and may still be waiting.'
            }
            if ($queueState -like '*not running' -and $status.phase -in @('training', 'exporting_validation', 'starting')) {
                Emit 'Status says active but queue PID is absent; inspect stderr/checkpoint before resuming.' 'WARN'
            }
            if ($status.error) { Emit "QUEUE FAILURE: $($status.error)" 'ERROR' }
            Show-Overview $runs $status $study.Label
            $lastStatuses[$study.Root] = $key
        }
        foreach ($run in $runs) {
            $state = "$($run.Summary.completed_epochs)|$($run.Exported)"
            $runKey = $run.Output
            if ($runStates.ContainsKey($runKey) -and $runStates[$runKey] -ne $state -and $run.Exported) {
                Emit "RUN COMPLETE + VALIDATION EXPORT: $($run.Name)" 'EVENT'
            }
            $runStates[$runKey] = $state
            # Track previous active runs too, so their final epoch is not lost at a switch.
            if ($run.Name -ne $status.current_run -and -not $epochSeen.ContainsKey($runKey)) { continue }
            if (-not $run.Summary) { continue }
            $csv = Join-Path $run.Output 'metrics.csv'
            if (-not (Test-Path $csv)) { continue }
            try {
                $savedRows = @(Import-Csv -LiteralPath $csv -Encoding UTF8 | Where-Object {
                    $_.epoch -match '^\d+$' -and [int]$_.epoch -le [int]$run.Summary.completed_epochs
                })
            } catch { continue }
            $rows = $savedRows
            $isHistory = -not $epochSeen.ContainsKey($runKey)
            if ($isHistory) {
                $epochSeen[$runKey] = 0
                $newEpochCounts[$runKey] = 0
                $rows = @($rows | Select-Object -Last $HistoryRows)
            }
            foreach ($m in $rows) {
                if ([int]$m.epoch -le $epochSeen[$runKey]) { continue }
                if ($lastMetricRun -ne $runKey) {
                    Show-MetricHeader $run
                    $lastMetricRun = $runKey
                }
                $seconds = [double]::Parse($m.train_seconds, $culture) + [double]::Parse($m.validation_seconds, $culture)
                $allocated = [double]::Parse($m.peak_allocated_bytes, $culture) / 1MB
                $reserved = [double]::Parse($m.peak_reserved_bytes, $culture) / 1MB
                $kind = if ($isHistory) { 'HISTORY' } else { 'EPOCH' }
                # Running best for THIS saved epoch, not a future best from summary.
                $best = ($savedRows | Where-Object { [int]$_.epoch -le [int]$m.epoch -and -not [string]::IsNullOrWhiteSpace($_.validation_accuracy) } | ForEach-Object { [double]::Parse($_.validation_accuracy, $culture) } | Measure-Object -Maximum).Maximum
                $bestText = Number $best
                Emit ('{0,-9} {1,9} {2,9} {3,9} {4,8} {5,9} {6,9}% {7,12}' -f `
                    "$($m.epoch)/$($run.Config.epochs)", (Number $m.train_loss 'F4'), (Number $m.train_weighted_accuracy), `
                    (Number $m.validation_loss 'F4'), (Number $m.validation_accuracy), $bestText, `
                    (Number (100.0 * [int]$m.epoch / $run.Config.epochs) 'F1'), (Number $seconds 'F1')) $kind
                if ($Details) { Emit "DETAIL: LR $(Number $m.lr_last 'G6') | AMPskip $($m.skipped_amp_steps) | GPU $(Number $allocated 'F0')/$(Number $reserved 'F0') MiB allocated/reserved" $kind }
                $epochSeen[$runKey] = [int]$m.epoch
                if (-not $isHistory) { $newEpochCounts[$runKey]++ }
            }
            if ($run.Name -eq $status.current_run -and $status.phase -eq 'training' -and $newEpochCounts[$runKey] -ge 5) {
                # Batch catch-up into one overview per poll; never count initial history.
                Emit "EXPERIMENT REMINDER: $($run.Name) | after 5 new saved epochs (batched if catching up)" 'EVENT'
                Show-Overview $runs $status $study.Label
                Show-MetricHeader $run
                $lastMetricRun = $runKey
                $newEpochCounts[$runKey] = $newEpochCounts[$runKey] % 5
            }
        }
    }
    # Follow grid stderr and development queue logs, not old smoke failures.
    $errorFiles = @()
    foreach ($entry in $configs) {
        $path = Join-Path $study.Root "logs\$($entry.Name).stderr.log"
        if (Test-Path $path) { $errorFiles += Get-Item $path }
    }
    $errorFiles += @(Get-ChildItem -Path (Join-Path $study.Root 'queue*.stderr.log') -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -notlike '*smoke*'
    })
    if ($ChainStatusPath -and $study -eq $studies[0]) {
        $chainStderr = Join-Path (Split-Path $ChainStatusPath -Parent) 'approved_chain.stderr.log'
        if (Test-Path -LiteralPath $chainStderr) { $errorFiles += Get-Item -LiteralPath $chainStderr }
    }
    foreach ($file in $errorFiles) {
        try {
            $raw = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
            $lines = @()
            if ($raw) {
                $parts = $raw -split "`n"
                # Leave an incomplete last line for the next poll.
                if ($parts.Count -gt 1) { $lines = @($parts[0..($parts.Count - 2)]) }
            }
            $firstRead = -not $errorSeen.ContainsKey($file.FullName)
            # A log CREATED after viewer startup is new, not old history.
            $historical = $firstRead -and $file.CreationTimeUtc -lt $sessionStarted
            if ($firstRead) { $errorSeen[$file.FullName] = 0 }
            if ($historical) {
                $errorSeen[$file.FullName] = $lines.Count
                if ($lines.Count -gt 0) {
                    Emit "HISTORICAL STDERR: $($file.Name) | $($lines.Count) saved lines (not a new failure). Details: $($file.FullName)" 'HISTORY'
                    if ($Details) { foreach ($line in ($lines | Select-Object -Last 10)) { Emit "HISTORY STDERR $($file.Name): $line" 'HISTORY' } }
                }
            }
            if ($lines.Count -lt $errorSeen[$file.FullName]) { $errorSeen[$file.FullName] = 0 }
            $offset = $errorSeen[$file.FullName]
            if ($offset -gt 0 -and $errorLast.ContainsKey($file.FullName) -and $lines[$offset - 1] -ne $errorLast[$file.FullName]) {
                $errorSeen[$file.FullName] = 0 # File truncated/replaced between polls.
            }
            $fresh = @($lines | Select-Object -Skip $errorSeen[$file.FullName] | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($fresh.Count -gt 0) {
                Emit "NEW STDERR: $($file.Name) | $($fresh.Count) new lines; check queue status. Details: $($file.FullName)" 'WARN'
                if ($Details) { foreach ($line in ($fresh | Select-Object -Last 10)) { Emit "STDERR $($file.Name): $($line.TrimEnd())" 'WARN' } }
            }
            $errorSeen[$file.FullName] = $lines.Count
            if ($lines.Count -gt 0) { $errorLast[$file.FullName] = $lines[-1] }
        } catch { } # Log may be replaced while reading; retry.
    }
    } # studies: continue following both roots through automatic chain transitions.
    $poll++
    if ($Once -or ($PollCount -gt 0 -and $poll -ge $PollCount)) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)