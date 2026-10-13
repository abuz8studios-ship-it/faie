# Judge chain — runs after the baseline perplexity finishes.
# Judges L40-tern (ARB-8/G2) and L40-r1024w (FACTOR-2b/G3) with flags IDENTICAL to
# the baseline run, then a temp-0 argmax check ("The capital of France is" -> " Paris").
# One judge at a time, by design (RAM/page-cache).
$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '0'
$rt = 'E:\ABU\MODELS\runtime\llama-b10069-cuda13'
$hold = 'E:\ABU\NEOCORTEX\corpus\calib_holdout.txt'
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$baseLog = 'E:\ABU\NEOCORTEX\ppl_baseline_gpu_ub4096_20260726_1042.log'

# 1. wait for the baseline run to finish (process gone AND Final estimate in its log)
while (Get-Process llama-perplexity -ErrorAction SilentlyContinue) { Start-Sleep 30 }
if (-not (Select-String -Path $baseLog -Pattern 'Final estimate' -Quiet)) {
    Write-Output "CHAIN ABORT: baseline log has no Final estimate — not judging patched models against a broken reference."
    exit 1
}
Write-Output "CHAIN: baseline complete, starting patched judges $(Get-Date -Format 'HH:mm:ss')"

$models = @(
    @{ name = 'tern';   path = 'F:\GLM52_FACTOR\L40-tern.gguf' },
    @{ name = 'r1024w'; path = 'F:\GLM52_FACTOR\L40-r1024w.gguf' }
)
foreach ($m in $models) {
    if (-not (Test-Path $m.path)) { Write-Output "CHAIN SKIP: $($m.path) missing"; continue }
    $plog = "E:\ABU\NEOCORTEX\ppl_$($m.name)_$stamp.log"
    Write-Output "CHAIN: judging $($m.name) -> $plog  $(Get-Date -Format 'HH:mm:ss')"
    & "$rt\llama-perplexity.exe" -m $m.path -f $hold -c 4096 -b 4096 -ub 4096 -ngl 99 `
        -ot 'ffn_.*_exps\.=CPU' -t 20 *> $plog
    $fin = Select-String -Path $plog -Pattern 'Final estimate' | Select-Object -Last 1
    Write-Output "CHAIN: $($m.name) PPL => $($fin.Line)"

    # temp-0 argmax oracle while the file is page-cache hot
    $olog = "E:\ABU\NEOCORTEX\oracle_$($m.name)_$stamp.log"
    & "$rt\llama-cli.exe" -m $m.path -ngl 99 -ot 'ffn_.*_exps\.=CPU' -t 20 `
        -p 'The capital of France is' -n 3 --temp 0 -no-cnv *> $olog
    $tail = (Get-Content $olog | Where-Object { $_ -match 'capital of France' } | Select-Object -Last 1)
    Write-Output "CHAIN: $($m.name) oracle => $tail"
}
Write-Output "CHAIN DONE $(Get-Date -Format 'HH:mm:ss')"
