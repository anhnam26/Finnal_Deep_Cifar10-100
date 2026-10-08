param(
    [string]$Python = "",
    [switch]$PreflightOnly
)

# Sequential, fail-stop transition: protected legacy -> approved new trials -> old pending jobs.
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
if (-not $Python) { $Python = (& python -B -c "import sys; print(sys.executable)").Trim() }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python interpreter missing: $Python" }

function Assert-Memory {
    $os = Get-CimInstance Win32_OperatingSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $ram = [math]::Round($os.FreePhysicalMemory / 1024)
    $commit = [math]::Round(($memory.CommitLimit - $memory.CommittedBytes) / 1MB)
    Write-Output "Free RAM: ${ram}MiB; free commit: ${commit}MiB"
    # Based on observed trainer ~3GiB commit plus startup/export safety margin.
    # Fail rather than repeatedly restart or change unrelated processes/pagefile.
    if ($ram -lt 1536 -or $commit -lt 5120) {
        throw "Insufficient headroom for guarded launch (need RAM>=1536MiB, commit>=5120MiB)."
    }
}

Assert-Memory
if ($PreflightOnly) { exit 0 }
$Output = Join-Path $Root 'outputs'
$Lock = Join-Path $Output 'approved_chain.lock'
$GpuLock = Join-Path $Output 'training.lock'
$Status = Join-Path $Output 'approved_chain_status.json'
if (Test-Path -LiteralPath $GpuLock) { throw "Another queue owns training.lock; inspect before retry." }
$stream = [System.IO.File]::Open($Lock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
$stream.Dispose()
$state = @{ pid=$PID; phase='starting'; test_evaluated=$false; started_utc=[DateTime]::UtcNow.ToString('o') }

function Save-State {
    $state['updated_utc'] = [DateTime]::UtcNow.ToString('o')
    $state | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Status -Encoding UTF8
}

try {
    @{pid=$PID; started_utc=$state['started_utc']} | ConvertTo-Json | Set-Content -LiteralPath $Lock -Encoding UTF8
    $Manifests = @(
        (Join-Path $Root 'configs\study\protected_resume.json'),
        (Join-Path $Root 'configs\improvement\manifest.json'),
        (Join-Path $Root 'configs\study\manifest.json')
    )
    foreach ($manifest in $Manifests) {
        Assert-Memory
        $state['phase'] = 'running_queue'
        $state['manifest'] = $manifest
        Save-State
        & $Python -B -u (Join-Path $Root 'study.py') --manifest $manifest --stage all
        if ($LASTEXITCODE -ne 0) { throw "Queue failed: $manifest (exit $LASTEXITCODE)" }
        $config = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
        $queueStatus = Get-Content -LiteralPath (Join-Path $config.output_root 'status.json') -Raw | ConvertFrom-Json
        if ($queueStatus.phase -ne 'grid_completed_awaiting_recipe_selection') {
            throw "Queue stopped without completing ($($queueStatus.phase)); chain stopped."
        }
    }
    $state['phase'] = 'completed_awaiting_validation_selection'
    Save-State
} catch {
    $state['phase'] = 'failed_or_blocked'
    $state['error'] = $_.Exception.Message
    Save-State
    throw
} finally {
    Remove-Item -LiteralPath $Lock -ErrorAction SilentlyContinue
}