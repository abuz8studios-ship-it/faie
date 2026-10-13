# ARB-8 / G2 JUDGE — does POST-HOC ternary preserve GLM-5.2?
#
# WHY THIS SCRIPT EXISTS AND THE OLD ONE DID NOT RUN:
#   judge_chain.ps1 waited on a baseline perplexity that never produced a
#   'Final estimate' (CPU run: 4h08m ETA, reached chunk 1; two GPU runs: nothing).
#   It correctly ABORTED rather than judge a patched model against a broken reference.
#   This script runs the BASELINE ITSELF first, with flags byte-identical to the
#   patched run, then judges. No baseline => no verdict. That is the whole discipline.
#
# WHAT IS BEING JUDGED:
#   F:\GLM52_FACTOR\L40-tern.gguf  -- layer 40's 128 experts ternarized in place
#   (BitNet absmean, group 128). Measured reconstruction error relF ~= 0.50 on all
#   three matrices, i.e. HALF the weight magnitude is wrong. If PPL barely moves, the
#   layer was redundant. If PPL explodes, post-hoc ternary is confirmed dead and
#   Bonsai-style QAT is the only ternary road -- which is the standing hypothesis.
#
# CONTROLS:
#   * identical flags, identical corpus, identical --chunks for both runs
#   * CUDA_VISIBLE_DEVICES=0 so Ornith on GPU1 (:8011, Hermes' only brain) is untouched
#   * experts forced to CPU (-ot) because the patched file is 146.9 GB
#   * --chunks 8 bounds wall time; 8 x 4096 = 32k tokens is far more than enough to
#     resolve a 50%-weight-error effect, and it is the SAME for both sides
#   * temp-0 argmax oracle on both: "The capital of France is" -> " Paris" (id 12089,
#     reference logprob -0.445 from llama.cpp b10069 on the unpatched model)
$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '0'

$rt    = 'E:\ABU\MODELS\runtime\llama-b10069-cuda13'
$hold  = 'E:\ABU\NEOCORTEX\corpus\calib_holdout.txt'
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$CHUNKS = 8

$runs = @(
  @{ name = 'BASELINE'; path = 'E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf' },
  @{ name = 'L40tern';  path = 'F:\GLM52_FACTOR\L40-tern.gguf' }
)

$results = @{}
foreach ($r in $runs) {
  if (-not (Test-Path $r.path)) { Write-Output "SKIP $($r.name): missing $($r.path)"; continue }
  $gb = [math]::Round((Get-Item $r.path).Length / 1GB, 1)
  Write-Output "=== $($r.name) START $(Get-Date -Format 'HH:mm:ss')  ($gb GiB)"

  $plog = "E:\ABU\NEOCORTEX\arb8_ppl_$($r.name)_$stamp.log"
  & "$rt\llama-perplexity.exe" -m $r.path -f $hold `
      -c 4096 -b 4096 -ub 4096 --chunks $CHUNKS `
      -ngl 99 -ot 'ffn_.*_exps\.=CPU' -t 20 *> $plog
  $ec = $LASTEXITCODE
  $fin = (Select-String -Path $plog -Pattern 'Final estimate' | Select-Object -Last 1).Line
  Write-Output "=== $($r.name) exit=$ec  $(Get-Date -Format 'HH:mm:ss')"
  Write-Output "PPL[$($r.name)] $fin"
  $results[$r.name] = $fin

  # oracle while the file is still page-cache warm
  $olog = "E:\ABU\NEOCORTEX\arb8_oracle_$($r.name)_$stamp.log"
  & "$rt\llama-cli.exe" -m $r.path -ngl 99 -ot 'ffn_.*_exps\.=CPU' -t 20 `
      -p 'The capital of France is' -n 4 --temp 0 -no-cnv *> $olog
  $orc = (Get-Content $olog -ErrorAction SilentlyContinue |
          Where-Object { $_ -match 'capital of France' } | Select-Object -Last 1)
  Write-Output "ORACLE[$($r.name)] $orc"
}

Write-Output "=== ARB8 SUMMARY ==="
foreach ($k in $results.Keys) { Write-Output "  $k : $($results[$k])" }
Write-Output "ARB8 JUDGE DONE $(Get-Date -Format 'HH:mm:ss')"
