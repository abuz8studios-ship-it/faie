# ARB-8 (G2) + FACTOR-2 (G3) JUDGE, v3 — fixes the thrash.
#
# WHAT v2 GOT WRONG:
#   v2 forced ALL experts to CPU (-ot 'ffn_.*_exps\.=CPU'), same as the baseline. But
#   reconstruct_layer.py writes the patched layer back as F16 (3.22 GB per tensor vs
#   Q2_K's 0.53), making the file 136.8 GiB against ~117 GiB usable RAM. With mmap on,
#   layer 40's 9.66 GB of F16 could never stay cached and was re-read from F: every
#   token, evicting everything else. Measured after 2h: 0 of 8 chunks done, RAM free
#   0.4 GB, CPU only 1,298s of ~7,300s wall = 18% => I/O bound, not compute bound.
#   llama.cpp even warned: "tensor overrides to CPU are used with mmap enabled".
#
# THE FIX: pin ONLY the patched layer's experts to GPU0 (9.66 GB into 28.7 GB free);
#   every other expert stays on CPU exactly as the baseline had it. Removes the hot
#   spot without changing anything else.
#
# SYNTAX TRAP: llama.cpp warns 'argument -ot specified multiple times ... only last
#   value will be used'. So this MUST be ONE comma-separated -ot, not two flags.
#
# BASELINE (already measured, do not redo):
#   PPL = 10.1192 +/- 0.23682  (8 chunks, all experts CPU, arb8_ppl_BASELINE_...2102.log)
#
# CAVEAT I AM NOT HIDING: blk.40 now computes on GPU where the baseline had it on CPU.
#   Float differences between CPU and CUDA kernels are ~1e-4 relative -- far below any
#   effect worth acting on. BUT if the measured delta comes back SMALL (<5%), that
#   caveat stops being negligible and a device-matched baseline MUST be run before the
#   result is trusted. Stated now so it cannot be rationalised later.
$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '0'          # Ornith owns GPU1 (:8011) -- untouched

$rt     = 'E:\ABU\MODELS\runtime\llama-b10069-cuda13'
$hold   = 'E:\ABU\NEOCORTEX\corpus\calib_holdout.txt'
$stamp  = Get-Date -Format 'yyyyMMdd_HHmm'
$CHUNKS = 8
$BASE   = 10.1192
$OT     = 'blk\.40\.ffn_.*_exps\.=CUDA0,ffn_.*_exps\.=CPU'

function Judge($name, $path) {
    if (-not (Test-Path $path)) { Write-Output "SKIP ${name}: missing $path"; return }
    $gb = [math]::Round((Get-Item $path).Length / 1GB, 1)
    Write-Output "=== $name START $(Get-Date -Format 'HH:mm:ss')  ($gb GiB)"
    $plog = "E:\ABU\NEOCORTEX\v3_ppl_${name}_$stamp.log"
    & "$rt\llama-perplexity.exe" -m $path -f $hold `
        -c 4096 -b 4096 -ub 4096 --chunks $CHUNKS `
        -ngl 99 -ot $OT -t 20 *> $plog
    $ec = $LASTEXITCODE
    $m = Select-String -Path $plog -Pattern 'Final estimate: PPL = ([0-9.]+)' | Select-Object -Last 1
    if ($m) {
        $ppl = [double]$m.Matches[0].Groups[1].Value
        $d = 100 * ($ppl - $BASE) / $BASE
        Write-Output ("RESULT[{0}] PPL = {1}  baseline {2}  delta {3:+0.0;-0.0}%  (x{4:N2})" -f `
                      $name, $ppl, $BASE, $d, ($ppl / $BASE))
    } else {
        Write-Output "RESULT[$name] NO FINAL ESTIMATE (exit=$ec) -- see $plog"
    }
}

Judge 'L40tern' 'F:\GLM52_FACTOR\L40-tern.gguf'

# --- G3 / FACTOR-2: existing r1024w is TRUNCATED (66.98 GB vs 146.91, unparseable,
#     build died at 45%, no stats.json). Rebuild -> parse-verify -> judge.
$out = 'F:\GLM52_FACTOR\L40-r1024w.gguf'
Write-Output "=== REBUILD r1024w START $(Get-Date -Format 'HH:mm:ss')  (tern took 1475s)"
Remove-Item $out -Force -ErrorAction SilentlyContinue
& python 'E:\ABU\NEOCORTEX\reconstruct_layer.py' `
    --layers 40 --mode rank --rank 1024 --metric imatrix --out $out `
    *> "E:\ABU\NEOCORTEX\v3_rebuild_r1024w_$stamp.log"
Write-Output "=== REBUILD exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss')"

$chk = & python -c @"
from gguf import GGUFReader
import os, sys
p = r'$out'
try:
    r = GGUFReader(p, 'r'); n = len(r.tensors)
    for t in r.tensors:
        if 'blk.40.ffn_gate_exps' in t.name:
            _ = t.data[0]
    print(f'OK tensors={n} gb={os.path.getsize(p)/1e9:.2f}')
except Exception as e:
    print(f'BAD {type(e).__name__}: {e}'); sys.exit(1)
"@ 2>&1
Write-Output "VERIFY: $chk"
if ("$chk" -notmatch 'OK tensors') {
    Write-Output 'ABORT: rebuilt r1024w does not parse -- refusing to judge a corrupt artifact.'
} else {
    Judge 'L40r1024w' $out
}
Write-Output "JUDGE v3 DONE $(Get-Date -Format 'HH:mm:ss')"
