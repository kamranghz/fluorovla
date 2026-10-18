# Supervised GR00T fine-tune with hang recovery + full-state resume (WSL2 GPU-PV VMA-lock hang workaround).
# AUTHORITATIVE PROGRESS SIGNAL = heartbeat file written from INSIDE the training process (launcher callback, FLUOROVLA_HEARTBEAT).
# docker logs are NEVER used for restart decisions (tqdm output lags). dxgkio_make_resident -12 is NEVER a failure signal.
# Restart only if heartbeat is stale beyond the limit AND independent evidence of a real stall exists
# (GPU idle for the whole recent window, kernel hung-task/D-state evidence, container gone, or docker unresponsive);
# a heartbeat stale > HbHardMin restarts regardless (a step takes ~20 s).
# Scientific config untouched: launch_catheter_finetune.py train /out/dataset /ckpt/run <name> <steps> 8 1e-4 <save_steps>.
param([string]$Name = "full600", [int]$Steps = 600, [int]$SaveSteps = 100, [int]$CopyFrom = 300, [int]$MaxAttempts = 40,
      [int]$HbStaleMin = 15, [int]$HbSaveStaleMin = 30, [int]$HbStartupMin = 30, [int]$HbHardMin = 60)
$ErrorActionPreference = "Continue"
$proj = "D:\projects\FluoroVLA"; $o = "$proj\runs\policy_training"
$sup = "$o\supervised_$Name"; New-Item -ItemType Directory -Force $sup, "$sup\evidence", "$o\ckpt_copies" | Out-Null
$log = "$sup\supervisor.log"; $hbHost = "$sup\heartbeat.json"; $hbCtr = "/out/supervised_$Name/heartbeat.json"
function Log($m) { $l = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; $l | Out-File $log -Append -Encoding utf8; Write-Output $l }
$env:MSYS_NO_PATHCONV = "1"
$mounts = @("-v", "D:\RoboticsResearch\repositories:/repo:ro", "-v", "${proj}:/proj:ro", "-v", "D:\RoboticsResearch\models:/models:ro",
            "-v", "D:\RoboticsResearch\caches\huggingface:/hf:ro", "-v", "${o}:/out", "-v", "gr00t_ckpt_full:/ckpt")
$runDir = "/ckpt/run/$Name"

function WithTimeout([scriptblock]$sb, [int]$sec, $argList = @()) {   # returns @{ok=;out=}; ok=$false if it did not finish (stack unresponsive)
    $j = Start-Job -ScriptBlock $sb -ArgumentList (,$argList)
    if (Wait-Job $j -Timeout $sec) { $out = Receive-Job $j; Remove-Job $j -Force; return @{ ok = $true; out = $out } }
    Stop-Job $j -ErrorAction SilentlyContinue; Remove-Job $j -Force -ErrorAction SilentlyContinue; return @{ ok = $false; out = $null }
}
function ProcOut([string]$exe, [string]$argline, [int]$sec) {   # run a native exe with a hard timeout; returns @{ok=;out=}
    $psi = New-Object System.Diagnostics.ProcessStartInfo; $psi.FileName = $exe; $psi.Arguments = $argline
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi); $so = $p.StandardOutput.ReadToEndAsync(); $null = $p.StandardError.ReadToEndAsync()
    if ($p.WaitForExit($sec * 1000)) { return @{ ok = $true; out = (($so.Result) -replace "`0", "") } }
    try { $p.Kill() } catch {}; return @{ ok = $false; out = "" }
}
function Helper([string]$cmd) {   # CPU-only helper container on the checkpoint volume
    $r = WithTimeout { param($a) $env:MSYS_NO_PATHCONV = "1"; docker run --rm -v gr00t_ckpt_full:/ckpt -v "$($a[0]):/proj:ro" -v "$($a[1]):/dst" --entrypoint sh gr00t:ecosystem -c $a[2] 2>&1 } 300 @($proj, "$o\ckpt_copies", $cmd)
    if ($r.ok) { return $r.out } else { return @() }
}
function DockerUp { $r = WithTimeout { & docker info *> $null; $LASTEXITCODE } 40; return ($r.ok -and $r.out -eq 0) }
function RecoverStackRaw([string]$why) {
    Log "RECOVER: $why -> docker kill, wsl --shutdown, restart Docker Desktop"
    WithTimeout { docker kill gr00t_train 2>&1 } 30 | Out-Null
    Get-Process -Name "Docker Desktop", "com.docker.backend" -ErrorAction SilentlyContinue | Stop-Process -Force
    wsl --shutdown; Start-Sleep 10
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    $up = $false
    foreach ($i in 1..72) { Start-Sleep 5; if (DockerUp) { $up = $true; break } }
    Start-Sleep 10
    Log "RECOVER: docker up=$up gpu_used_MiB=$((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim())"
    return $up
}
function EnsureWarm {   # never launch on a young VM: wait until it is >= 5 min old, run a trivial CUDA op, require clean exit + clean kernel log
    for ($try = 1; $try -le 2; $try++) {
        $r = ProcOut "wsl.exe" "-d docker-desktop -e sh -c `"cut -d. -f1 /proc/uptime`"" 60
        $up = 0; if ($r.ok) { $m = [regex]::Match($r.out, "(?m)^\s*(\d+)\s*$"); if ($m.Success) { $up = [int]$m.Groups[1].Value } }
        if ($up -lt 300) { $w = 300 - $up; Log "WARMUP: VM uptime ${up}s < 300s -> waiting ${w}s"; Start-Sleep $w }
        $w = WithTimeout { param($a) $env:MSYS_NO_PATHCONV = "1"; docker run --rm --gpus all -v "$($a[0]):/proj:ro" --entrypoint /opt/gr00t-venv/bin/python gr00t:ecosystem /proj/tools/groot/cuda_warmup.py 2>&1; "rc=$LASTEXITCODE" } 240 @(,$proj)
        $ke = KernelEvidence
        $ok = $w.ok -and (($w.out | Out-String) -match "warmup ok") -and ($w.out -contains "rc=0") -and (-not $ke.hung)
        Log "WARMUP try ${try}: ok=$ok kernel_hung=$($ke.hung) gpu=$((nvidia-smi --query-gpu=memory.used --format=csv,noheader).Trim())"
        if ($ok) { return $true }
        Log "WARMUP failed -> stack recovery"; if (-not (RecoverStackRaw "warmup failed")) { return $false }
    }
    return $false
}
function RecoverStack([string]$why) { return (RecoverStackRaw $why) }
function MemInfo { # VM MemAvailable/AnonPages/Swap (GB) + container RSS (GB); $null fields if unavailable
    $r = ProcOut "wsl.exe" "-d docker-desktop -e sh -c `"grep -E '^(MemAvailable|AnonPages|SwapTotal|SwapFree):' /proc/meminfo; echo MI_DONE`"" 40
    $o = @{ avail = $null; anon = $null; swapUsed = $null; rss = $null }
    if ($r.ok -and $r.out -match "MI_DONE") {
        foreach ($k in "MemAvailable", "AnonPages", "SwapTotal", "SwapFree") { $m = [regex]::Match($r.out, "${k}:\s+(\d+)"); if ($m.Success) { Set-Variable -Name "v$k" -Value ([double]$m.Groups[1].Value / 1MB) } }
        $o.avail = [math]::Round($vMemAvailable, 1); $o.anon = [math]::Round($vAnonPages, 1); $o.swapUsed = [math]::Round($vSwapTotal - $vSwapFree, 1)
    }
    $d = WithTimeout { docker stats --no-stream --format "{{.MemUsage}}" gr00t_train 2>&1 } 40
    if ($d.ok) { $m = [regex]::Match(($d.out | Out-String), "([\d.]+)GiB"); if ($m.Success) { $o.rss = [double]$m.Groups[1].Value } }
    return $o
}
function KernelEvidence { # hashtable: responsive, hung (dmesg hung-task / coredump block / D-state python|cuda), text
    $cmd = "dmesg | grep -E 'blocked for|Blocked by coredump|blocked on an rw-semaphore|__vma_start_write' | tail -4; ps -eo stat,comm | grep -E '^D.* (python|cuda)' | head -3; echo KE_DONE"
    $r = ProcOut "wsl.exe" ("-d docker-desktop -e sh -c `"$cmd`"") 60
    if (-not $r.ok -or $r.out -notmatch "KE_DONE") { return @{ responsive = $false; hung = $false; text = "wsl query failed/timed out: $($r.out)" } }
    $body = ($r.out -split "KE_DONE")[0]
    return @{ responsive = $true; hung = ($body -match "blocked for|Blocked by coredump|rw-semaphore|(?m)^D\S*\s+(python|cuda)"); text = $body.Trim() }
}
function Evidence([int]$n, [string]$why) {
    $d = "$sup\evidence\attempt$n"; New-Item -ItemType Directory -Force $d | Out-Null
    "$why" | Out-File "$d\reason.txt" -Encoding utf8
    if (Test-Path $hbHost) { Copy-Item $hbHost "$d\heartbeat_last.json" -Force }
    (KernelEvidence).text | Out-File "$d\kernel.txt" -Encoding utf8
    (nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used --format=csv) | Out-File "$d\gpu.txt" -Encoding utf8
    $r = WithTimeout { docker logs --tail 40 gr00t_train 2>&1 } 30; if ($r.ok) { $r.out | Out-File "$d\docker_tail_informational.txt" -Encoding utf8 }
}
$script:ckFail = $false
function Checkpoints {   # a FAILED listing (helper container could not run, e.g. host mounts lost) must never be read as "no checkpoints"
    $out = @(Helper "python3 /proj/tools/groot/validate_checkpoint.py $runDir $($args -join ' '); echo __LISTING_OK__")
    if (-not ($out -contains "__LISTING_OK__")) { $script:ckFail = $true; return @() }
    $script:ckFail = $false
    return @($out | Where-Object { $_ -match "^\d+ " })
}
function MaxValid($lines) { $v = @($lines | Where-Object { $_ -match "^\d+ VALID" } | ForEach-Object { [int](($_ -split " ")[0]) }); if ($v.Count) { return ($v | Measure-Object -Maximum).Maximum } else { return 0 } }
function CopyDone {
    foreach ($l in (Checkpoints)) {
        $p = $l -split " "; if ($p[1] -ne "VALID") { continue }; $n = [int]$p[0]
        if ($n -lt $CopyFrom -or (Test-Path "$o\ckpt_copies\$Name-checkpoint-$n")) { continue }
        Log "COPY checkpoint-$n -> D: (weights only)"
        $r = Helper "set -e; d=/dst/$Name-checkpoint-$n.partial; rm -rf `$d; mkdir -p `$d; cd $runDir/checkpoint-$n; cp -a model-*.safetensors model.safetensors.index.json config.json trainer_state.json training_args.bin `$d/; mv `$d /dst/$Name-checkpoint-$n; echo copied"
        Log "COPY checkpoint-$n result: $($r | Select-Object -Last 1)"
        if ($n -eq $Steps) {
            Log "FINAL ARTIFACTS: full checkpoint-$n -> D: and loss history"
            $fr = Helper "set -e; d=/dst/$Name-checkpoint-$n-FULL; rm -rf `$d.partial; cp -a $runDir/checkpoint-$n `$d.partial; mv `$d.partial `$d; ls -l `$d | head -30; echo FULLCOPY_OK"
            Log "FULL checkpoint-$n copy: $($fr | Select-Object -Last 1)"
            $lh = Helper "python3 /proj/tools/groot/export_loss_history.py $runDir/checkpoint-$n /dst/$Name-loss_history.json"
            Log "loss history export: $($lh | Select-Object -Last 1)"
        }
    }
}
function ReadHb { if (-not (Test-Path $hbHost)) { return $null }; try { return (Get-Content $hbHost -Raw | ConvertFrom-Json) } catch { return $null } }

$noProgress = 0; $lowMem = 0; $resumeStalls = 0
for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    if (-not (DockerUp)) { if (-not (RecoverStack "docker not up at attempt $attempt")) { Log "ABORT: docker will not start"; exit 2 } }
    $cks = Checkpoints "--delete-invalid"
    if ($script:ckFail) {
        Log "CKPT listing FAILED (docker/host mounts unusable) -> stack recovery before any launch"
        if (-not (RecoverStack "checkpoint listing failed")) { Log "ABORT: docker will not start"; exit 2 }
        $cks = Checkpoints "--delete-invalid"
        if ($script:ckFail) { Log "ABORT: checkpoint listing still failing after stack recovery - will NOT launch (would risk restarting from step 0)"; exit 5 }
    }
    foreach ($l in $cks) { Log "CKPT $l" }
    $resumeFrom = MaxValid $cks
    if ($resumeFrom -ge $Steps) { Log "DONE: checkpoint-$resumeFrom already valid"; break }
    if (-not (EnsureWarm)) { Log "ABORT: warm-up could not be completed cleanly - NOT launching (local WSL2 unreliable)"; exit 6 }
    WithTimeout { docker rm -f gr00t_train *> $null } 60 | Out-Null
    Remove-Item $hbHost -Force -ErrorAction SilentlyContinue
    $envs = @("-e", "NO_ALBUMENTATIONS_UPDATE=1", "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "FLUOROVLA_SAVE_TOTAL_LIMIT=2", "-e", "FLUOROVLA_HEARTBEAT=$hbCtr")
    if ($resumeFrom -gt 0) { $envs += @("-e", "FLUOROVLA_RESUME=1", "-e", "MALLOC_MMAP_MAX_=0", "-e", "MALLOC_TRIM_THRESHOLD_=68719476736") }   # resumed process only: glibc never returns freed memory to the kernel
    Log "ATTEMPT $attempt start: resume_from_step=$resumeFrom resume=$($resumeFrom -gt 0) baseline_gpu_MiB=$((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim())"
    docker run -d --name gr00t_train --gpus all --shm-size=8g --entrypoint /opt/gr00t-venv/bin/python @envs @mounts gr00t:ecosystem `
        /proj/src/fluorovla/policy/finetune_launcher.py train /out/dataset /ckpt/run $Name $Steps 8 1e-4 $SaveSteps | Out-Null
    $tStart = Get-Date; $lastSeq = -1; $lastHbChange = $tStart; $hb = $null; $lastProg = $tStart; $lastNote = Get-Date "2000-01-01"
    $gpuHist = New-Object System.Collections.ArrayList; $stall = $null; $everBeat = $false; $stepAtLastProg = -1; $tAtLastProg = $tStart; $lastValidSeen = $resumeFrom
    while ($true) {
        Start-Sleep 30
        $now = Get-Date
        $hb = ReadHb
        if ($hb -and $hb.seq -ne $lastSeq) { $lastSeq = $hb.seq; $lastHbChange = $now; $everBeat = $true }
        $util = [int]((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -replace "[^0-9]", "")
        [void]$gpuHist.Add(@{ t = $now; u = $util }); while ($gpuHist.Count -gt 0 -and ($now - $gpuHist[0].t).TotalMinutes -gt 8) { $gpuHist.RemoveAt(0) }
        $step = if ($hb) { [int]$hb.global_step } else { -1 }; $phase = if ($hb) { $hb.phase } else { "no_heartbeat_yet" }
        if (($now - $lastProg).TotalMinutes -ge 10) {
            $rate = if ($step -gt $stepAtLastProg -and $stepAtLastProg -ge 0) { "{0:N1}s/step" -f (($now - $tAtLastProg).TotalSeconds / ($step - $stepAtLastProg)) } else { "n/a" }
            $stepAtLastProg = $step; $tAtLastProg = $now; $lastProg = $now
            $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
            $mi = MemInfo
            if ($mi.avail -ne $null -and $mi.avail -lt 1.5) { $lowMem++ } else { $lowMem = 0 }
            if ($lowMem -ge 2) { Log "ABORT: MEMORY_UNSAFE vm_available=$($mi.avail)GB swap_used=$($mi.swapUsed)GB container_rss=$($mi.rss)GB (2 consecutive checks < 1.5 GB) - supervisor stops, container left running for inspection"; Evidence $attempt "memory unsafe"; exit 7 }
            Log "MEM vm_available_GB=$($mi.avail) vm_anon_GB=$($mi.anon) swap_used_GB=$($mi.swapUsed) container_mem_GB=$($mi.rss)"
            Log "HEARTBEAT attempt=$attempt step=$step phase=$phase hb_age_s=$([int](($now - $lastHbChange).TotalSeconds)) rate=$rate gpu=$((nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used --format=csv,noheader).Trim()) hostFreeGB=$free"
            CopyDone
            $mv = MaxValid (Checkpoints); if ($mv -gt $lastValidSeen) { $lastValidSeen = $mv; Log "CKPT newest valid = checkpoint-$mv" }
        }
        # ---- terminal conditions ----
        if ($hb -and ($hb.phase -eq "train_end" -or $step -ge $Steps)) {
            if ((Checkpoints | Where-Object { $_ -match "^$Steps VALID" }).Count -gt 0) { Log "FINAL checkpoint-$Steps VALID (heartbeat phase=$phase step=$step)"; $stall = "final"; break }
        }
        $r = WithTimeout { docker ps -q --filter name=gr00t_train } 40
        if ($r.ok -and -not $r.out) { Log "CONTAINER EXITED (heartbeat step=$step phase=$phase)"; $stall = "exited"; break }
        # ---- stall decision: stale heartbeat AND independent evidence ----
        $stale = ($now - $lastHbChange).TotalMinutes
        $limit = if (-not $everBeat) { $HbStartupMin } elseif ($phase -eq "saving") { $HbSaveStaleMin } else { $HbStaleMin }
        if ($stale -gt $limit) {
            $gpuIdle = ($gpuHist.Count -ge 10) -and (($gpuHist | ForEach-Object { $_.u } | Measure-Object -Maximum).Maximum -lt 30)
            $ke = KernelEvidence
            $dockerDead = -not $r.ok
            $why = @(); if ($gpuIdle) { $why += "GPU idle (<30% for last 8 min)" }; if ($ke.hung) { $why += "kernel hung-task/D-state" }; if ($dockerDead) { $why += "docker unresponsive" }; if (-not $ke.responsive) { $why += "wsl unresponsive" }
            if ($why.Count -gt 0) { $stall = ("STALL: heartbeat stale {0:N0} min (limit $limit, step $step, phase $phase) + evidence: {1}" -f $stale, ($why -join "; ")); break }
            if ($stale -gt $HbHardMin) { $stall = ("STALL(hard cap): heartbeat stale {0:N0} min > $HbHardMin (step $step, phase $phase), GPU busy but no progress" -f $stale); break }
            if (($now - $lastNote).TotalMinutes -ge 10) { $lastNote = $now; Log ("NOTE heartbeat stale {0:N0} min (step $step, phase $phase) but GPU busy / no kernel evidence -> NOT restarting" -f $stale) }
        }
    }
    Evidence $attempt $stall
    if ($resumeFrom -gt 0 -and $stall -like "STALL*" -and $step -le $resumeFrom) {
        $resumeStalls++
        Log "RESUME_STALL #$resumeStalls of 3 from checkpoint-$resumeFrom (resume did not pass its start step; VMA/D-state evidence in evidencettempt$attempt)"
        if ($resumeStalls -ge 3) {
            Log "ABORT: 3 clean local resume attempts from checkpoint-$resumeFrom all stalled before passing step $resumeFrom - per operator rule STOP and report (no automatic cloud migration). checkpoint-$resumeFrom preserved."
            exit 8
        }
    } elseif ($step -gt $resumeFrom) { $resumeStalls = 0 }
    CopyDone
    if ((Checkpoints | Where-Object { $_ -match "^$Steps VALID" }).Count -gt 0) {
        $fl = WithTimeout { docker logs gr00t_train 2>&1 } 120; if ($fl.ok) { $fl.out | Out-File "$supinal_container_log_last_attempt.txt" -Encoding utf8 }
        Log "TRAINING COMPLETE; releasing container (kill + stack restart in case of exit hang)"
        RecoverStack "post-run cleanup" | Out-Null; CopyDone; Log "SUPERVISOR DONE"; exit 0
    }
    $newBest = MaxValid (Checkpoints)
    if ($newBest -le $resumeFrom) { $noProgress++ } else { $noProgress = 0 }
    Log "ATTEMPT $attempt ended ($stall); valid checkpoint step $resumeFrom -> $newBest; consecutive no-progress attempts=$noProgress"
    if ($noProgress -ge 6) { Log "ABORT: 6 consecutive attempts without a new valid checkpoint - needs human attention"; exit 3 }
    if ($stall -like "STALL*" -or -not (DockerUp)) { RecoverStack $stall | Out-Null }
}
Log "ABORT: max attempts reached"; exit 4
