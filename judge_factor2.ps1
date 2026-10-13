# FACTOR-2 / G3 JUDGE — does a rank-1024 reconstruction keep behaviour?
#
# WHY THERE IS A REBUILD STEP:
#   The existing F:\GLM52_FACTOR\L40-r1024w.gguf is TRUNCATED and unreadable. Verified
#   2026-07-29: gguf-py raises "cannot reshape array of size 0 into shape (128,6144)";
#   the file is 66.98 GB against tern's 146.91 GB; its build log's last line is
#   "Writing: 45%|65.7G/147G"; it never printed "done in" and never wrote its
#   .stats.json. Judging it would have produced a confident garbage number.
#   So: rebuild first, verify it PARSES, and only then judge.
#
# EXPECTED OUTCOME, stated before measuring:
#   The interrupted build DID log its reconstruction error before dying --
#   relF mean 0.8613 / 0.8765 / 0.8805, kept imatrix energy 0.2608 / 0.2917 / 0.2862.
#   So r=1024 discards ~71-74% of the data-metric energy of layer 40, versus ternary's
#   relF ~= 0.50. FACTOR-1 already killed factorization on the spectrum; this converts
#   that structural verdict into a behavioural one. Predict: WORSE than L40-tern.
#
# CHAINED, not concurrent: waits for the ARB-8 judge to finish. Two llama-perplexity
# processes (or a 147 GB write racing a 147 GB read on F:) would thrash both.
# Flags are byte-identical to judge_arb8.ps1 so the SAME baseline is a valid reference.
$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '0'

$rt    = 'E:\ABU\MODELS\runtime\llama-b10069-cuda13'
$hold  = 'E:\ABU\NEOCORTEX\corpus\calib_holdout.txt'
$out   = 'F:\GLM52_FACTOR\L40-r1024w.gguf'
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$CHUNKS = 8

# 1. wait for the ARB-8 judge (perplexity or cli) to clear
Write-Output "WAIT: for ARB-8 judge to finish  $(Get-Date -Format 'HH:mm:ss')"
while ((Get-Process llama-perplexity -ErrorAction SilentlyContinue) -or
       (Get-Process llama-cli -ErrorAction SilentlyContinue)) { Start-Sleep 30 }
Write-Output "CLEAR at $(Get-Date -Format 'HH:mm:ss')"

# 2. rebuild the truncated artifact
Write-Output "REBUILD start $(Get-Date -Format 'HH:mm:ss')  (tern took 1475s)"
Remove-Item $out -Force -ErrorAction SilentlyContinue
$blog = "E:\ABU\NEOCORTEX\factor2_rebuild_$stamp.log"
& python 'E:\ABU\NEOCORTEX\reconstruct_layer.py' `
    --layers 40 --mode rank --rank 1024 --metric imatrix --out $out *> $blog
Write-Output "REBUILD exit=$LASTEXITCODE  $(Get-Date -Format 'HH:mm:ss')"

# 3. VERIFY IT PARSES before judging -- the whole reason this script exists
$chk = & python -c @"
from gguf import GGUFReader
import os,sys
p=r'$out'
try:
    r=GGUFReader(p,'r'); n=len(r.tensors)
    for t in r.tensors:
        if 'blk.40.ffn_gate_exps' in t.name:
            _ = t.data[0]
    print(f'OK tensors={n} bytes={os.path.getsize(p)}')
except Exception as e:
    print(f'BAD {type(e).__name__}: {e}'); sys.exit(1)
"@ 2>&1
Write-Output "VERIFY: $chk"
if ($chk -notmatch '^OK') {
    Write-Output "ABORT: rebuilt file still does not parse -- not judging a corrupt artifact."
    exit 1
}

# 4. judge, identical flags to ARB-8
$plog = "E:\ABU\NEOCORTEX\factor2_ppl_r1024w_$stamp.log"
Write-Output "JUDGE start $(Get-Date -Format 'HH:mm:ss')"
& "$rt\llama-perplexity.exe" -m $out -f $hold `
    -c 4096 -b 4096 -ub 4096 --chunks $CHUNKS `
    -ngl 99 -ot 'ffn_.*_exps\.=CPU' -t 20 *> $plog
$fin = (Select-String -Path $plog -Pattern 'Final estimate' | Select-Object -Last 1).Line
Write-Output "PPL[r1024w] $fin"

$olog = "E:\ABU\NEOCORTEX\factor2_oracle_r1024w_$stamp.log"
& "$rt\llama-cli.exe" -m $out -ngl 99 -ot 'ffn_.*_exps\.=CPU' -t 20 `
    -p 'The capital of France is' -n 4 --temp 0 -no-cnv *> $olog
$orc = (Get-Content $olog -ErrorAction SilentlyContinue |
        Where-Object { $_ -match 'capital of France' } | Select-Object -Last 1)
Write-Output "ORACLE[r1024w] $orc"
Write-Output "FACTOR2 JUDGE DONE $(Get-Date -Format 'HH:mm:ss')"
