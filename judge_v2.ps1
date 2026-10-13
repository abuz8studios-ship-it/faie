# ARB-8 (G2) + FACTOR-2 (G3) JUDGE, v2 — perplexity only.
#
# BASELINE IS ALREADY MEASURED, do not redo it:
#   GLM-5.2-REAM-Q2_K unpatched, calib_holdout, 8 chunks, 88.91 s/pass
#   PPL = 10.1192 +/- 0.23682     (arb8_ppl_BASELINE_20260729_2102.log, exit 0)
#
# WHY v1's ORACLE IS DELETED, not fixed:
#   v1 ran `llama-cli -p 'The capital of France is' -n 4 --temp 0`. GLM-5.2 is a
#   REASONING model: it spent all 4 tokens inside [Start thinking] ("The user wants
#   me"), emitted no answer, then sat at the interactive '>' prompt waiting on stdin
#   that never comes -- 42 minutes hung, blocking the whole chain, at 1.8 tok/s with
#   experts on CPU. This is the identical trap already written into CLAUDE.md at 07:49
#   the same day for Ornith ("returns EMPTY content if max_tokens is small; a smoke
#   test with a low max_tokens would report a live brain as broken"). A single-token
#   argmax probe is the wrong instrument for a thinking model.
#   Perplexity is the right instrument and it is quantitative: +/-0.237 CI on 8 chunks
#   resolves any effect worth acting on. The Paris oracle was a FAIE-era tool from when
#   no PPL harness existed. Dropped, not patched.
#
# FLAGS ARE BYTE-IDENTICAL to the baseline run, or the comparison is void.
$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '0'          # Ornith owns GPU1 (:8011) -- do not touch

$rt     = 'E:\ABU\MODELS\runtime\llama-b10069-cuda13'
$hold   = 'E:\ABU\NEOCORTEX\corpus\calib_holdout.txt'
$stamp  = Get-Date -Format 'yyyyMMdd_HHmm'
$CHUNKS = 8
$BASE   = 10.1192

function Judge($name, $path) {
    if (-not (Test-Path $path)) { Write-Output "SKIP ${name}: missing $path"; return }
    $gb = [math]::Round((Get-Item $path).Length / 1GB, 1)
    Write-Output "=== $name START $(Get-Date -Format 'HH:mm:ss')  ($gb GiB)"
    $plog = "E:\ABU\NEOCORTEX\v2_ppl_${name}_$stamp.log"
    & "$rt\llama-perplexity.exe" -m $path -f $hold `
        -c 4096 -b 4096 -ub 4096 --chunks $CHUNKS `
        -ngl 99 -ot 'ffn_.*_exps\.=CPU' -t 20 *> $plog
    $ec = $LASTEXITCODE
    $m = Select-String -Path $plog -Pattern 'Final estimate: PPL = ([0-9.]+)' |
         Select-Object -Last 1
    if ($m) {
        $ppl = [double]$m.Matches[0].Groups[1].Value
        $d = 100 * ($ppl - $BASE) / $BASE
        Write-Output ("RESULT[{0}] PPL = {1}  vs baseline {2}  delta {3:+0.0;-0.0}%  (x{4:N2})" -f `
                      $name, $ppl, $BASE, $d, ($ppl / $BASE))
    } else {
        Write-Output "RESULT[$name] NO FINAL ESTIMATE (exit=$ec) -- see $plog"
    }
}

# --- G2 / ARB-8 : post-hoc ternary, relF ~= 0.50 -----------------------------
Judge 'L40tern' 'F:\GLM52_FACTOR\L40-tern.gguf'

# --- G3 / FACTOR-2 : rank-1024 whitened. Existing file is TRUNCATED (66.98 GB
#     vs 146.91, unparseable, build died at 45%). Rebuild -> verify -> judge.
$out = 'F:\GLM52_FACTOR\L40-r1024w.gguf'
Write-Output "=== REBUILD r1024w START $(Get-Date -Format 'HH:mm:ss')  (tern took 1475s)"
Remove-Item $out -Force -ErrorAction SilentlyContinue
$blog = "E:\ABU\NEOCORTEX\v2_rebuild_r1024w_$stamp.log"
& python 'E:\ABU\NEOCORTEX\reconstruct_layer.py' `
    --layers 40 --mode rank --rank 1024 --metric imatrix --out $out *> $blog
Write-Output "=== REBUILD exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss')"

$chk = & python -c @"
from gguf import GGUFReader
import os, sys
p = r'$out'
try:
    r = GGUFReader(p, 'r')
    n = len(r.tensors)
    for t in r.tensors:
        if 'blk.40.ffn_gate_exps' in t.name:
            _ = t.data[0]          # force a real read of patched data
    print(f'OK tensors={n} gb={os.path.getsize(p)/1e9:.2f}')
except Exception as e:
    print(f'BAD {type(e).__name__}: {e}'); sys.exit(1)
"@ 2>&1
Write-Output "VERIFY: $chk"
if ("$chk" -notmatch 'OK tensors') {
    Write-Output 'ABORT: rebuilt r1024w still does not parse -- refusing to judge a corrupt artifact.'
} else {
    Judge 'L40r1024w' $out
}

Write-Output "JUDGE v2 DONE $(Get-Date -Format 'HH:mm:ss')"
