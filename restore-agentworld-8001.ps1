# Restore the Qwen-AgentWorld-35B-A3B swap that was on GPU1 :8001.
# Killed 2026-07-26 to free 11,272 MiB for FAIE expert residency (Ahmad authorized).
# Exact command line captured from PID 32668 before termination.
& "E:\ABU\MODELS\runtime\llama-b10069-cuda13\llama-server.exe" `
    -m "E:\ABU\MODELS\llm\qwen-agentworld-35b-a3b\Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf" `
    --host 127.0.0.1 --port 8001 -mg 0 -c 32768
