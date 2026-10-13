// FAIE Inference Kernels — Ternary + Q2_K + Q3_K compressed matmul
// Arabic compression: read quantized weights directly on GPU, no FP16 expansion
// Standalone: nvcc -shared -O3 --use_fast_math -arch=sm_120 -Xcompiler "/MD" -o ternary_kernel.dll ternary_kernel.cu

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>

extern "C" {

// Ternary matvec: y = W_ternary * x
// packed_W: [out_features * in_features/4] bytes, 4 ternary values per byte
// Packing: byte = (v0+1) | ((v1+1)<<2) | ((v2+1)<<4) | ((v3+1)<<6)
// To unpack: (byte >> (i*2)) & 3 - 1 gives {-1, 0, +1}
// scales: [out_features * groups_per_row] float32, one per group_size values
__global__ void ternary_matvec_kernel(
    const uint8_t* __restrict__ packed_W,
    const float*   __restrict__ scales,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int out_features,
    int in_features,
    int group_size
) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    int packed_per_row = in_features / 4;
    int groups_per_row = (in_features + group_size - 1) / group_size;

    const uint8_t* row_packed = packed_W + (long long)row * packed_per_row;
    const float* row_scales = scales + (long long)row * groups_per_row;

    float acc = 0.0f;

    for (int j = threadIdx.x; j < packed_per_row; j += blockDim.x) {
        uint8_t p = row_packed[j];
        int base = j * 4;

        int v0 = (int)(p & 0x03) - 1;
        int v1 = (int)((p >> 2) & 0x03) - 1;
        int v2 = (int)((p >> 4) & 0x03) - 1;
        int v3 = (int)((p >> 6) & 0x03) - 1;

        int grp = base / group_size;
        float s = row_scales[grp];

        float dot = (float)v0 * x[base]
                  + (float)v1 * x[base + 1]
                  + (float)v2 * x[base + 2]
                  + (float)v3 * x[base + 3];
        acc += s * dot;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Block reduction via shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        acc = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) y[row] = acc;
    }
}

// Expert forward: SwiGLU = down(SiLU(gate(x)) * up(x))
// Phase 0: gate_buf = SiLU(gate_packed * x)
// Phase 1: up_buf = up_packed * x
// Phase 2: y = down_packed * (gate_buf * up_buf)

// Shared reduction helper (used by all phases)
__device__ float block_reduce(float acc, float* ws) {
    for (int off = 16; off > 0; off /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) >> 5;
        acc = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off /= 2)
            acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    return acc;
}

// Ternary matvec core: one row of packed_W * x
__device__ float ternary_dot(
    const uint8_t* __restrict__ row_packed,
    const float*   __restrict__ row_scales,
    const float*   __restrict__ x,
    int packed_per_row, int group_size
) {
    float acc = 0.0f;
    for (int j = threadIdx.x; j < packed_per_row; j += blockDim.x) {
        uint8_t p = row_packed[j];
        int base = j * 4;
        int v0 = (int)(p & 0x03) - 1;
        int v1 = (int)((p >> 2) & 0x03) - 1;
        int v2 = (int)((p >> 4) & 0x03) - 1;
        int v3 = (int)((p >> 6) & 0x03) - 1;
        float s = row_scales[base / group_size];
        acc += s * ((float)v0*x[base] + (float)v1*x[base+1]
                   + (float)v2*x[base+2] + (float)v3*x[base+3]);
    }
    return acc;
}

__global__ void ternary_expert_phase0_kernel(
    const uint8_t* __restrict__ gate_packed,
    const float*   __restrict__ gate_scales,
    const float*   __restrict__ x,
    float*         __restrict__ gate_buf,
    int gate_out,
    int gate_in,
    int group_size
) {
    int row = blockIdx.x;
    if (row >= gate_out) return;

    int packed_per_row = gate_in / 4;
    int groups_per_row = (gate_in + group_size - 1) / group_size;

    float acc = ternary_dot(
        gate_packed + (long long)row * packed_per_row,
        gate_scales + (long long)row * groups_per_row,
        x, packed_per_row, group_size);

    __shared__ float ws[32];
    acc = block_reduce(acc, ws);

    if (threadIdx.x == 0) {
        float sig = 1.0f / (1.0f + expf(-acc));
        gate_buf[row] = acc * sig;
    }
}

__global__ void ternary_expert_phase1_up_kernel(
    const uint8_t* __restrict__ up_packed,
    const float*   __restrict__ up_scales,
    const float*   __restrict__ x,
    float*         __restrict__ up_buf,
    int up_out,
    int up_in,
    int group_size
) {
    int row = blockIdx.x;
    if (row >= up_out) return;

    int packed_per_row = up_in / 4;
    int groups_per_row = (up_in + group_size - 1) / group_size;

    float acc = ternary_dot(
        up_packed + (long long)row * packed_per_row,
        up_scales + (long long)row * groups_per_row,
        x, packed_per_row, group_size);

    __shared__ float ws[32];
    acc = block_reduce(acc, ws);

    if (threadIdx.x == 0) up_buf[row] = acc;
}

__global__ void ternary_expert_phase2_down_kernel(
    const uint8_t* __restrict__ down_packed,
    const float*   __restrict__ down_scales,
    const float*   __restrict__ gate_buf,
    const float*   __restrict__ up_buf,
    float*         __restrict__ y,
    int down_out,
    int down_in,
    int group_size
) {
    int row = blockIdx.x;
    if (row >= down_out) return;

    int packed_per_row = down_in / 4;
    int groups_per_row = (down_in + group_size - 1) / group_size;

    const uint8_t* rp = down_packed + (long long)row * packed_per_row;
    const float* rs = down_scales + (long long)row * groups_per_row;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < packed_per_row; j += blockDim.x) {
        uint8_t p = rp[j];
        int base = j * 4;
        int v0 = (int)(p & 0x03) - 1;
        int v1 = (int)((p >> 2) & 0x03) - 1;
        int v2 = (int)((p >> 4) & 0x03) - 1;
        int v3 = (int)((p >> 6) & 0x03) - 1;
        float s = rs[base / group_size];
        // Read SwiGLU hidden: gate_buf * up_buf (elementwise)
        float h0 = gate_buf[base]     * up_buf[base];
        float h1 = gate_buf[base + 1] * up_buf[base + 1];
        float h2 = gate_buf[base + 2] * up_buf[base + 2];
        float h3 = gate_buf[base + 3] * up_buf[base + 3];
        acc += s * ((float)v0*h0 + (float)v1*h1 + (float)v2*h2 + (float)v3*h3);
    }

    __shared__ float ws[32];
    acc = block_reduce(acc, ws);

    if (threadIdx.x == 0) y[row] = acc;
}


// C-exported host functions (ctypes interface)
__declspec(dllexport) void ternary_matvec(
    const uint8_t* packed_W, const float* scales, const float* x, float* y,
    int out_features, int in_features, int group_size
) {
    int threads = 256;
    ternary_matvec_kernel<<<out_features, threads>>>(
        packed_W, scales, x, y, out_features, in_features, group_size
    );
}

__declspec(dllexport) void ternary_expert_forward(
    const uint8_t* gate_packed, const float* gate_scales,
    const uint8_t* up_packed,   const float* up_scales,
    const uint8_t* down_packed, const float* down_scales,
    const float* x, float* gate_buf, float* up_buf, float* y,
    int gate_out, int gate_in, int down_out, int down_in, int group_size
) {
    int threads = 256;
    ternary_expert_phase0_kernel<<<gate_out, threads>>>(
        gate_packed, gate_scales, x, gate_buf, gate_out, gate_in, group_size
    );
    ternary_expert_phase1_up_kernel<<<gate_out, threads>>>(
        up_packed, up_scales, x, up_buf, gate_out, gate_in, group_size
    );
    ternary_expert_phase2_down_kernel<<<down_out, threads>>>(
        down_packed, down_scales, gate_buf, up_buf, y, down_out, down_in, group_size
    );
}

// ============================================================
// Q2_K Compressed Matvec: y = W_q2k * x
// Reads Q2_K quantized weights DIRECTLY on GPU — no dequant step.
// Block layout (84 bytes per 256 elements):
//   [0:16]  scales_and_mins (16 uint8)
//   [16:80] qs (64 bytes = 256 x 2-bit quants)
//   [80:82] d (float16 super-scale)
//   [82:84] dmin (float16 super-min)
// ============================================================
__global__ void q2k_matvec_kernel(
    const uint8_t* __restrict__ W_data,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int n_rows,
    int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int blocks_per_row = (in_features + 255) / 256;
    int bytes_per_row = blocks_per_row * 84;
    const uint8_t* row_data = W_data + (long long)row * bytes_per_row;

    float acc = 0.0f;
    int total_qs = in_features / 4;

    for (int qi = threadIdx.x; qi < total_qs; qi += blockDim.x) {
        int elem = qi * 4;
        int blk = elem / 256;
        int pos = elem % 256;
        int sub = pos / 16;

        const uint8_t* block = row_data + blk * 84;
        float d   = __half2float(*(const __half*)(block + 80));
        float dm  = __half2float(*(const __half*)(block + 82));
        uint8_t sc = block[sub];
        float dl = d  * (float)(sc & 0x0F);
        float ml = dm * (float)(sc >> 4);

        uint8_t qb = block[16 + pos / 4];
        float v0 = dl * (float)( qb       & 3) - ml;
        float v1 = dl * (float)((qb >> 2) & 3) - ml;
        float v2 = dl * (float)((qb >> 4) & 3) - ml;
        float v3 = dl * (float)((qb >> 6) & 3) - ml;

        acc += v0*x[elem] + v1*x[elem+1] + v2*x[elem+2] + v3*x[elem+3];
    }

    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) y[row] = acc;
}

// ============================================================
// Q3_K Compressed Matvec: y = W_q3k * x
// Block layout (110 bytes per 256 elements):
//   [0:32]   hmask (32 bytes = 256 high bits)
//   [32:96]  qs (64 bytes = 256 x 2-bit low quants)
//   [96:108] scales (12 bytes → 16 x 6-bit signed)
//   [108:110] d (float16 super-scale)
// ============================================================
__device__ void q3k_unpack_scales(const uint8_t* sc_raw, float* sc_out) {
    for (int i = 0; i < 8; i++) {
        int s = sc_raw[i] & 0x3F;
        if (s >= 32) s -= 64;
        sc_out[i] = (float)s;
    }
    for (int i = 0; i < 4; i++) {
        int lo = sc_raw[8 + i] & 0x0F;
        int hi = (sc_raw[8 + i] >> 4) & 0x0F;
        int s1 = lo | ((sc_raw[i] >> 6) << 4);
        int s2 = hi | ((sc_raw[i + 4] >> 6) << 4);
        if (s1 >= 32) s1 -= 64;
        if (s2 >= 32) s2 -= 64;
        sc_out[8 + i]  = (float)s1;
        sc_out[12 + i] = (float)s2;
    }
}

// Single-scale extractor: same result as q3k_unpack_scales(sc)[j] but WITHOUT
// materializing a float[16]. The array version forces the compiler to place it in
// local memory (= global DRAM traffic) because `sc[sub]` is a dynamic index.
__device__ __forceinline__ float q3k_scale_at(const uint8_t* __restrict__ sc, int j) {
    int s;
    if (j < 8) {
        s = sc[j] & 0x3F;
    } else if (j < 12) {
        int i = j - 8;
        s = (sc[8 + i] & 0x0F) | ((sc[i] >> 6) << 4);
    } else {
        int i = j - 12;
        s = ((sc[8 + i] >> 4) & 0x0F) | ((sc[i + 4] >> 6) << 4);
    }
    return (float)(s >= 32 ? s - 64 : s);
}

__global__ void q3k_matvec_kernel(
    const uint8_t* __restrict__ W_data,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int n_rows,
    int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int blocks_per_row = (in_features + 255) / 256;
    int bytes_per_row = blocks_per_row * 110;
    const uint8_t* row_data = W_data + (long long)row * bytes_per_row;

    float acc = 0.0f;
    int total_qs = in_features / 4;

    for (int qi = threadIdx.x; qi < total_qs; qi += blockDim.x) {
        int elem = qi * 4;
        int blk = elem / 256;
        int pos = elem % 256;
        int sub = pos / 16;

        const uint8_t* block = row_data + blk * 110;
        float d = __half2float(*(const __half*)(block + 108));

        float scales[16];
        q3k_unpack_scales(block + 96, scales);
        float dl = d * scales[sub];

        uint8_t qb = block[32 + pos / 4];
        int q0 = ( qb       & 3);
        int q1 = ((qb >> 2) & 3);
        int q2 = ((qb >> 4) & 3);
        int q3 = ((qb >> 6) & 3);

        int h0 = (block[pos / 8] >> (pos % 8)) & 1;
        int h1 = (block[(pos+1) / 8] >> ((pos+1) % 8)) & 1;
        int h2 = (block[(pos+2) / 8] >> ((pos+2) % 8)) & 1;
        int h3 = (block[(pos+3) / 8] >> ((pos+3) % 8)) & 1;

        float v0 = dl * (float)(q0 - 4 + h0 * 4);
        float v1 = dl * (float)(q1 - 4 + h1 * 4);
        float v2 = dl * (float)(q2 - 4 + h2 * 4);
        float v3 = dl * (float)(q3 - 4 + h3 * 4);

        acc += v0*x[elem] + v1*x[elem+1] + v2*x[elem+2] + v3*x[elem+3];
    }

    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) y[row] = acc;
}

// Host exports
__declspec(dllexport) void q2k_matvec(
    const uint8_t* W, const float* x, float* y, int n_rows, int in_features
) {
    q2k_matvec_kernel<<<n_rows, 256>>>(W, x, y, n_rows, in_features);
}

__declspec(dllexport) void q3k_matvec(
    const uint8_t* W, const float* x, float* y, int n_rows, int in_features
) {
    q3k_matvec_kernel<<<n_rows, 256>>>(W, x, y, n_rows, in_features);
}

// SiLU(gate) * up elementwise
__global__ void swiglu_fuse_kernel(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

// Fused SwiGLU expert: gate(Q2K) + up(Q2K) + SiLU*mul + down(Q3K) in ONE host call
__declspec(dllexport) void expert_swiglu(
    const uint8_t* gate_W, const uint8_t* up_W, const uint8_t* down_W,
    const float* x, float* gate_buf, float* up_buf, float* y,
    int gate_rows, int gate_in,
    int down_rows, int down_in,
    int gate_qtype, int down_qtype
) {
    // Phase 1: gate matvec
    if (gate_qtype == 0)
        q2k_matvec_kernel<<<gate_rows, 256>>>(gate_W, x, gate_buf, gate_rows, gate_in);
    else
        q3k_matvec_kernel<<<gate_rows, 256>>>(gate_W, x, gate_buf, gate_rows, gate_in);

    // Phase 2: up matvec (same dims as gate)
    if (gate_qtype == 0)
        q2k_matvec_kernel<<<gate_rows, 256>>>(up_W, x, up_buf, gate_rows, gate_in);
    else
        q3k_matvec_kernel<<<gate_rows, 256>>>(up_W, x, up_buf, gate_rows, gate_in);

    // Phase 3: SiLU(gate) * up → gate_buf
    int blocks = (gate_rows + 255) / 256;
    swiglu_fuse_kernel<<<blocks, 256>>>(gate_buf, up_buf, gate_rows);

    // Phase 4: down matvec
    if (down_qtype == 0)
        q2k_matvec_kernel<<<down_rows, 256>>>(down_W, gate_buf, y, down_rows, down_in);
    else
        q3k_matvec_kernel<<<down_rows, 256>>>(down_W, gate_buf, y, down_rows, down_in);
}

// Fused attention: 4 matvecs in ONE host call (q_a, q_b, kv_a, out)
__declspec(dllexport) void attention_matvecs(
    const uint8_t* qa_W, const uint8_t* qb_W,
    const uint8_t* kva_W, const uint8_t* out_W,
    const float* x,
    float* qa_buf, float* qb_buf, float* kva_buf, float* out_buf,
    int qa_rows, int qa_in, int qa_qt,
    int qb_rows, int qb_in, int qb_qt,
    int kva_rows, int kva_in, int kva_qt,
    int out_rows, int out_in, int out_qt
) {
    // Q path: x -> q_a
    if (qa_qt == 0)
        q2k_matvec_kernel<<<qa_rows, 256>>>(qa_W, x, qa_buf, qa_rows, qa_in);
    else
        q3k_matvec_kernel<<<qa_rows, 256>>>(qa_W, x, qa_buf, qa_rows, qa_in);

    // KV path: x -> kv_a (independent of Q, can run in parallel on GPU)
    if (kva_qt == 0)
        q2k_matvec_kernel<<<kva_rows, 256>>>(kva_W, x, kva_buf, kva_rows, kva_in);
    else
        q3k_matvec_kernel<<<kva_rows, 256>>>(kva_W, x, kva_buf, kva_rows, kva_in);
}

// RMS norm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += 256)
        sum_sq += x[i] * x[i];
    smem[tid] = sum_sq;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // GGUF glm-dsa attention.layer_norm_rms_epsilon = 1e-5 (was hardcoded 1e-6)
    float rms = rsqrtf(smem[0] / (float)n + 1e-5f);
    for (int i = tid; i < n; i += 256)
        out[i] = x[i] * rms * w[i];
}

// y += weight * src elementwise
__global__ void weighted_add_kernel(float* y, const float* src, float weight, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += weight * src[i];
}

// Zero a buffer
__global__ void zero_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

// FP16 matvec: y = W_fp16 * x_fp32 (for router: small matrix, FP16 weights)
__global__ void fp16_matvec_kernel(const __half* __restrict__ W, const float* __restrict__ x,
                                    float* __restrict__ y, int n_rows, int in_features) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int j = tid; j < in_features; j += 256)
        sum += __half2float(W[row * in_features + j]) * x[j];
    smem[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}

// TopK on GPU: find top-K values + indices from an array of n floats
// glm-dsa.expert_weights_scale from the GGUF. Routed-expert outputs are multiplied
// by this after renormalization. Runtime-settable so it can be A/B'd against the
// residual-norm growth rather than trusted blindly.
__device__ float d_router_scale = 2.5f;
static float h_router_scale = 2.5f;
__declspec(dllexport) void set_router_scale(float s) {
    h_router_scale = s;
    cudaMemcpyToSymbol(d_router_scale, &s, sizeof(float));
}
__declspec(dllexport) float get_router_scale() { return h_router_scale; }
#define ROUTER_WEIGHT_SCALE d_router_scale

__global__ void topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                             float* __restrict__ topk_vals, int* __restrict__ topk_ids,
                             int n, int k) {
    // Single-block kernel: 1 block, 256 threads
    // Each thread holds k best (insertion sort)
    __shared__ float s_vals[8];  // max k=8
    __shared__ int s_ids[8];

    // glm-dsa / DeepSeek-V3 routing, per the GGUF metadata:
    //   expert_gating_func = 2      -> SIGMOID scores (not softmax)
    //   expert_weights_norm = true  -> renormalize the chosen k
    //   expert_weights_scale = 2.5  -> then multiply
    // The bias participates in SELECTION ONLY; the weight is the raw sigmoid score.
    // The old code softmaxed (logit+bias) over the top-k, which got all three wrong
    // and corrupted every MoE layer's mixture and magnitude.
    __shared__ float s_sel[8];
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) { s_sel[i] = -1e30f; s_vals[i] = 0.0f; s_ids[i] = 0; }
        for (int i = 0; i < n; i++) {
            float sc  = 1.0f / (1.0f + __expf(-logits[i]));      // sigmoid gate
            float sel = sc + (bias ? bias[i] : 0.0f);            // selection key
            if (sel > s_sel[k-1]) {
                s_sel[k-1] = sel; s_vals[k-1] = sc; s_ids[k-1] = i;
                for (int j = k-2; j >= 0; j--) {
                    if (s_sel[j+1] > s_sel[j]) {
                        float ts = s_sel[j];  s_sel[j]  = s_sel[j+1];  s_sel[j+1]  = ts;
                        float tv = s_vals[j]; s_vals[j] = s_vals[j+1]; s_vals[j+1] = tv;
                        int   ti = s_ids[j];  s_ids[j]  = s_ids[j+1];  s_ids[j+1]  = ti;
                    }
                }
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < k; i++) sum += s_vals[i];
        float inv = (sum > 1e-20f) ? (1.0f / sum) : 1.0f;
        for (int i = 0; i < k; i++) {
            topk_vals[i] = s_vals[i] * inv * ROUTER_WEIGHT_SCALE;
            topk_ids[i]  = s_ids[i];
        }
    }
}

// Batched expert SwiGLU: process N experts in ONE host call
// gate_ptrs/up_ptrs/down_ptrs are HOST arrays of DEVICE pointers
// weights is a HOST array of N float softmax weights
__declspec(dllexport) void batched_expert_swiglu(
    const void** gate_ptrs, const void** up_ptrs, const void** down_ptrs,
    const float* weights,  // HOST: N softmax weights
    const float* x,        // device: input vector
    float* gate_buf, float* up_buf, float* y,  // device: scratch + output
    float* temp_buf,       // device: per-expert down output (size HIDDEN_DIM)
    int n_experts,
    int gate_rows, int gate_in,
    int down_rows, int down_in,
    int gate_qtype, int down_qtype
) {
    // Zero output
    zero_kernel<<<(down_rows + 255) / 256, 256>>>(y, down_rows);

    for (int e = 0; e < n_experts; e++) {
        const uint8_t* gW = (const uint8_t*)gate_ptrs[e];
        const uint8_t* uW = (const uint8_t*)up_ptrs[e];
        const uint8_t* dW = (const uint8_t*)down_ptrs[e];

        if (gate_qtype == 0)
            q2k_matvec_kernel<<<gate_rows, 256>>>(gW, x, gate_buf, gate_rows, gate_in);
        else
            q3k_matvec_kernel<<<gate_rows, 256>>>(gW, x, gate_buf, gate_rows, gate_in);

        if (gate_qtype == 0)
            q2k_matvec_kernel<<<gate_rows, 256>>>(uW, x, up_buf, gate_rows, gate_in);
        else
            q3k_matvec_kernel<<<gate_rows, 256>>>(uW, x, up_buf, gate_rows, gate_in);

        swiglu_fuse_kernel<<<(gate_rows + 255) / 256, 256>>>(gate_buf, up_buf, gate_rows);

        if (down_qtype == 0)
            q2k_matvec_kernel<<<down_rows, 256>>>(dW, gate_buf, temp_buf, down_rows, down_in);
        else
            q3k_matvec_kernel<<<down_rows, 256>>>(dW, gate_buf, temp_buf, down_rows, down_in);

        weighted_add_kernel<<<(down_rows + 255) / 256, 256>>>(y, temp_buf, weights[e], down_rows);
    }
}

// Fused MoE layer: norm → router → topK → shared_exp → 8×experts → residual
__declspec(dllexport) void fused_moe_layer(
    // Input/output
    float* x, int hidden_dim,
    // Attention weights
    const uint8_t* qa_W, int qa_rows, int qa_in, int qa_qt,
    const uint8_t* qb_W, int qb_rows, int qb_in, int qb_qt,
    const uint8_t* kva_W, int kva_rows, int kva_in, int kva_qt,
    const uint8_t* out_W, int out_rows, int out_in, int out_qt,
    // Attention norms
    const float* attn_norm_w, const float* q_norm_w, const float* kv_norm_w,
    // FFN norm
    const float* ffn_norm_w,
    // Router
    const __half* router_W, int n_router_experts, const float* expert_bias,
    // Shared expert
    const uint8_t* shexp_gate, const uint8_t* shexp_up, const uint8_t* shexp_down,
    int shexp_gate_rows, int shexp_gate_in, int shexp_down_rows, int shexp_down_in,
    int shexp_gate_qt, int shexp_down_qt,
    // Expert pointers (host arrays of device ptrs)
    const void** expert_gate_ptrs, const void** expert_up_ptrs, const void** expert_down_ptrs,
    int exp_gate_rows, int exp_gate_in, int exp_down_rows, int exp_down_in,
    int exp_gate_qt, int exp_down_qt,
    // Scratch buffers (all device)
    float* norm_buf, float* qa_buf, float* qb_buf, float* kva_buf, float* attn_out_buf,
    float* gate_buf, float* up_buf, float* expert_out_buf, float* temp_buf,
    float* router_logits, float* topk_vals, int* topk_ids,
    // Config
    int top_k
) {
    // === ATTENTION ===
    // 1. RMS norm
    rms_norm_kernel<<<1, 256>>>(x, attn_norm_w, norm_buf, hidden_dim);

    // 2. q_a matvec
    if (qa_qt == 0)
        q2k_matvec_kernel<<<qa_rows, 256>>>((const uint8_t*)qa_W, norm_buf, qa_buf, qa_rows, qa_in);
    else
        q3k_matvec_kernel<<<qa_rows, 256>>>((const uint8_t*)qa_W, norm_buf, qa_buf, qa_rows, qa_in);

    // 3. q_a norm
    rms_norm_kernel<<<1, 256>>>(qa_buf, q_norm_w, qa_buf, qa_rows);

    // 4. q_b matvec
    if (qb_qt == 0)
        q2k_matvec_kernel<<<qb_rows, 256>>>((const uint8_t*)qb_W, qa_buf, qb_buf, qb_rows, qb_in);
    else
        q3k_matvec_kernel<<<qb_rows, 256>>>((const uint8_t*)qb_W, qa_buf, qb_buf, qb_rows, qb_in);

    // 5. kv_a matvec
    if (kva_qt == 0)
        q2k_matvec_kernel<<<kva_rows, 256>>>((const uint8_t*)kva_W, norm_buf, kva_buf, kva_rows, kva_in);
    else
        q3k_matvec_kernel<<<kva_rows, 256>>>((const uint8_t*)kva_W, norm_buf, kva_buf, kva_rows, kva_in);

    // 6. output projection (simplified: skip actual attention, just project q)
    if (out_qt == 0)
        q2k_matvec_kernel<<<out_rows, 256>>>((const uint8_t*)out_W, qb_buf, attn_out_buf, out_rows, out_in);
    else
        q3k_matvec_kernel<<<out_rows, 256>>>((const uint8_t*)out_W, qb_buf, attn_out_buf, out_rows, out_in);

    // 7. Residual add: x += attn_out (convert fp32 add)
    weighted_add_kernel<<<(hidden_dim + 255) / 256, 256>>>(x, attn_out_buf, 1.0f, hidden_dim);

    // === MoE FFN ===
    // 8. FFN RMS norm
    rms_norm_kernel<<<1, 256>>>(x, ffn_norm_w, norm_buf, hidden_dim);

    // 9. Router: FP16 matvec
    fp16_matvec_kernel<<<n_router_experts, 256>>>(router_W, norm_buf, router_logits, n_router_experts, hidden_dim);

    // 10. TopK + softmax on GPU
    topk_kernel<<<1, 1>>>(router_logits, expert_bias, topk_vals, topk_ids, n_router_experts, top_k);

    // 11. Shared expert SwiGLU
    if (shexp_gate_qt == 0)
        q2k_matvec_kernel<<<shexp_gate_rows, 256>>>(shexp_gate, norm_buf, gate_buf, shexp_gate_rows, shexp_gate_in);
    else
        q3k_matvec_kernel<<<shexp_gate_rows, 256>>>(shexp_gate, norm_buf, gate_buf, shexp_gate_rows, shexp_gate_in);

    if (shexp_gate_qt == 0)
        q2k_matvec_kernel<<<shexp_gate_rows, 256>>>(shexp_up, norm_buf, up_buf, shexp_gate_rows, shexp_gate_in);
    else
        q3k_matvec_kernel<<<shexp_gate_rows, 256>>>(shexp_up, norm_buf, up_buf, shexp_gate_rows, shexp_gate_in);

    swiglu_fuse_kernel<<<(shexp_gate_rows + 255) / 256, 256>>>(gate_buf, up_buf, shexp_gate_rows);

    if (shexp_down_qt == 0)
        q2k_matvec_kernel<<<shexp_down_rows, 256>>>(shexp_down, gate_buf, expert_out_buf, shexp_down_rows, shexp_down_in);
    else
        q3k_matvec_kernel<<<shexp_down_rows, 256>>>(shexp_down, gate_buf, expert_out_buf, shexp_down_rows, shexp_down_in);

    // expert_out_buf now has shared expert result — add to x later

    // 12. Sync to read topk_ids on host (needed for expert pointer lookup)
    cudaDeviceSynchronize();

    int h_topk_ids[8];
    float h_topk_vals[8];
    cudaMemcpy(h_topk_ids, topk_ids, top_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_topk_vals, topk_vals, top_k * sizeof(float), cudaMemcpyDeviceToHost);

    // 13. Top-K experts
    for (int e = 0; e < top_k; e++) {
        int eidx = h_topk_ids[e];
        const uint8_t* gW = (const uint8_t*)expert_gate_ptrs[eidx];
        const uint8_t* uW = (const uint8_t*)expert_up_ptrs[eidx];
        const uint8_t* dW = (const uint8_t*)expert_down_ptrs[eidx];

        if (exp_gate_qt == 0)
            q2k_matvec_kernel<<<exp_gate_rows, 256>>>(gW, norm_buf, gate_buf, exp_gate_rows, exp_gate_in);
        else
            q3k_matvec_kernel<<<exp_gate_rows, 256>>>(gW, norm_buf, gate_buf, exp_gate_rows, exp_gate_in);

        if (exp_gate_qt == 0)
            q2k_matvec_kernel<<<exp_gate_rows, 256>>>(uW, norm_buf, up_buf, exp_gate_rows, exp_gate_in);
        else
            q3k_matvec_kernel<<<exp_gate_rows, 256>>>(uW, norm_buf, up_buf, exp_gate_rows, exp_gate_in);

        swiglu_fuse_kernel<<<(exp_gate_rows + 255) / 256, 256>>>(gate_buf, up_buf, exp_gate_rows);

        if (exp_down_qt == 0)
            q2k_matvec_kernel<<<exp_down_rows, 256>>>(dW, gate_buf, temp_buf, exp_down_rows, exp_down_in);
        else
            q3k_matvec_kernel<<<exp_down_rows, 256>>>(dW, gate_buf, temp_buf, exp_down_rows, exp_down_in);

        weighted_add_kernel<<<(exp_down_rows + 255) / 256, 256>>>(expert_out_buf, temp_buf, h_topk_vals[e], exp_down_rows);
    }

    // 14. Add expert result to x (residual)
    weighted_add_kernel<<<(hidden_dim + 255) / 256, 256>>>(x, expert_out_buf, 1.0f, hidden_dim);
}

// Fused gate+up+SiLU: reads BOTH gate and up Q2K weights in one pass
// out[row] = SiLU(gate_W[row].x) * up_W[row].x
__global__ void q2k_gate_up_silu_kernel(
    const uint8_t* __restrict__ gate_W,
    const uint8_t* __restrict__ up_W,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    int n_rows, int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int bpr = (in_features + 255) / 256;
    int byte_row = bpr * 84;
    const uint8_t* gr = gate_W + (long long)row * byte_row;
    const uint8_t* ur = up_W  + (long long)row * byte_row;
    float ga = 0.0f, ua = 0.0f;
    int tq = in_features / 4;
    for (int qi = threadIdx.x; qi < tq; qi += blockDim.x) {
        int e = qi * 4, blk = e / 256, pos = e % 256, sub = pos / 16, qo = 16 + pos/4;
        const uint8_t* gb = gr + blk*84;
        float gd = __half2float(*(const __half*)(gb+80));
        float gm = __half2float(*(const __half*)(gb+82));
        float gdl = gd*(float)(gb[sub]&0x0F), gml = gm*(float)(gb[sub]>>4);
        uint8_t gq = gb[qo];
        const uint8_t* ub = ur + blk*84;
        float ud = __half2float(*(const __half*)(ub+80));
        float um = __half2float(*(const __half*)(ub+82));
        float udl = ud*(float)(ub[sub]&0x0F), uml = um*(float)(ub[sub]>>4);
        uint8_t uq = ub[qo];
        float x0=x[e], x1=x[e+1], x2=x[e+2], x3=x[e+3];
        ga += (gdl*(float)(gq&3)-gml)*x0 + (gdl*(float)((gq>>2)&3)-gml)*x1
            + (gdl*(float)((gq>>4)&3)-gml)*x2 + (gdl*(float)((gq>>6)&3)-gml)*x3;
        ua += (udl*(float)(uq&3)-uml)*x0 + (udl*(float)((uq>>2)&3)-uml)*x1
            + (udl*(float)((uq>>4)&3)-uml)*x2 + (udl*(float)((uq>>6)&3)-uml)*x3;
    }
    for (int off = 16; off > 0; off /= 2) {
        ga += __shfl_down_sync(0xffffffff, ga, off);
        ua += __shfl_down_sync(0xffffffff, ua, off);
    }
    __shared__ float ws[64];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { ws[wid] = ga; ws[32+wid] = ua; }
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x+31)>>5;
        ga = (lane<nw)?ws[lane]:0.0f;
        ua = (lane<nw)?ws[32+lane]:0.0f;
        for (int off = 16; off > 0; off /= 2) {
            ga += __shfl_down_sync(0xffffffff, ga, off);
            ua += __shfl_down_sync(0xffffffff, ua, off);
        }
        if (lane == 0) { float sig = 1.0f/(1.0f+expf(-ga)); out[row] = ga*sig*ua; }
    }
}

// Down matvec + weighted accumulate: y[row] += weight * (W_q2k[row] . hidden)
__global__ void q2k_down_accum_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ h,
    float* __restrict__ y, float weight, int n_rows, int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int bpr = (in_features+255)/256;
    const uint8_t* rd = W + (long long)row * bpr * 84;
    float acc = 0.0f;
    int tq = in_features/4;
    for (int qi = threadIdx.x; qi < tq; qi += blockDim.x) {
        int e=qi*4, blk=e/256, pos=e%256, sub=pos/16;
        const uint8_t* b = rd+blk*84;
        float d=__half2float(*(const __half*)(b+80));
        float dm=__half2float(*(const __half*)(b+82));
        float dl=d*(float)(b[sub]&0x0F), ml=dm*(float)(b[sub]>>4);
        uint8_t qb=b[16+pos/4];
        acc += (dl*(float)(qb&3)-ml)*h[e] + (dl*(float)((qb>>2)&3)-ml)*h[e+1]
             + (dl*(float)((qb>>4)&3)-ml)*h[e+2] + (dl*(float)((qb>>6)&3)-ml)*h[e+3];
    }
    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) y[row] += weight * acc;
}

// Q3K version of down+accumulate
__global__ void q3k_down_accum_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ h,
    float* __restrict__ y, float weight, int n_rows, int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int bpr = (in_features+255)/256;
    const uint8_t* rd = W + (long long)row * bpr * 110;
    float acc = 0.0f;
    int tq = in_features/4;
    for (int qi = threadIdx.x; qi < tq; qi += blockDim.x) {
        int e=qi*4, blk=e/256, pos=e%256, sub=pos/16;
        const uint8_t* b = rd+blk*110;
        float d = __half2float(*(const __half*)(b+108));
        float sc[16]; q3k_unpack_scales(b+96, sc);
        float dl = d * sc[sub];
        uint8_t qb = b[32+pos/4];
        int q0=(qb&3),q1=((qb>>2)&3),q2=((qb>>4)&3),q3=((qb>>6)&3);
        int h0=(b[pos/8]>>(pos%8))&1, h1=(b[(pos+1)/8]>>((pos+1)%8))&1;
        int h2=(b[(pos+2)/8]>>((pos+2)%8))&1, h3=(b[(pos+3)/8]>>((pos+3)%8))&1;
        acc += dl*(float)(q0-4+h0*4)*h[e] + dl*(float)(q1-4+h1*4)*h[e+1]
             + dl*(float)(q2-4+h2*4)*h[e+2] + dl*(float)(q3-4+h3*4)*h[e+3];
    }
    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) y[row] += weight * acc;
}

// MEGA gate+up+SiLU: ALL experts in ONE launch. 2D grid: (n_rows, n_experts)
// Pointers passed directly as kernel args — ZERO cudaMemcpyAsync
__global__ void mega8_q2k_gate_up_silu_kernel(
    const uint8_t* g0, const uint8_t* g1, const uint8_t* g2, const uint8_t* g3,
    const uint8_t* g4, const uint8_t* g5, const uint8_t* g6, const uint8_t* g7,
    const uint8_t* u0, const uint8_t* u1, const uint8_t* u2, const uint8_t* u3,
    const uint8_t* u4, const uint8_t* u5, const uint8_t* u6, const uint8_t* u7,
    const float* __restrict__ x, float* __restrict__ out,
    int n_rows, int in_features
) {
    int expert = blockIdx.y, row = blockIdx.x;
    const uint8_t* gW, *uW;
    switch(expert) {
        case 0: gW=g0; uW=u0; break; case 1: gW=g1; uW=u1; break;
        case 2: gW=g2; uW=u2; break; case 3: gW=g3; uW=u3; break;
        case 4: gW=g4; uW=u4; break; case 5: gW=g5; uW=u5; break;
        case 6: gW=g6; uW=u6; break; default: gW=g7; uW=u7; break;
    }
    int bpr = (in_features+255)/256, byte_row = bpr*84;
    const uint8_t* gr = gW + (long long)row*byte_row;
    const uint8_t* ur = uW + (long long)row*byte_row;
    float ga=0.0f, ua=0.0f;
    int tq = in_features/4;
    for (int qi = threadIdx.x; qi < tq; qi += blockDim.x) {
        int e=qi*4, blk=e/256, pos=e%256, sub=pos/16, qo=16+pos/4;
        const uint8_t* gb=gr+blk*84;
        float gd=__half2float(*(const __half*)(gb+80)), gm=__half2float(*(const __half*)(gb+82));
        float gdl=gd*(float)(gb[sub]&0x0F), gml=gm*(float)(gb[sub]>>4);
        uint8_t gq=gb[qo];
        const uint8_t* ub=ur+blk*84;
        float ud=__half2float(*(const __half*)(ub+80)), um=__half2float(*(const __half*)(ub+82));
        float udl=ud*(float)(ub[sub]&0x0F), uml=um*(float)(ub[sub]>>4);
        uint8_t uq=ub[qo];
        float x0=x[e],x1=x[e+1],x2=x[e+2],x3=x[e+3];
        ga+=(gdl*(float)(gq&3)-gml)*x0+(gdl*(float)((gq>>2)&3)-gml)*x1
           +(gdl*(float)((gq>>4)&3)-gml)*x2+(gdl*(float)((gq>>6)&3)-gml)*x3;
        ua+=(udl*(float)(uq&3)-uml)*x0+(udl*(float)((uq>>2)&3)-uml)*x1
           +(udl*(float)((uq>>4)&3)-uml)*x2+(udl*(float)((uq>>6)&3)-uml)*x3;
    }
    for (int off=16;off>0;off/=2){ga+=__shfl_down_sync(0xffffffff,ga,off);ua+=__shfl_down_sync(0xffffffff,ua,off);}
    __shared__ float ws[64];
    int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    if(lane==0){ws[wid]=ga;ws[32+wid]=ua;}
    __syncthreads();
    if(wid==0){
        int nw=(blockDim.x+31)>>5;
        ga=(lane<nw)?ws[lane]:0.0f; ua=(lane<nw)?ws[32+lane]:0.0f;
        for(int off=16;off>0;off/=2){ga+=__shfl_down_sync(0xffffffff,ga,off);ua+=__shfl_down_sync(0xffffffff,ua,off);}
        if(lane==0){float sig=1.0f/(1.0f+expf(-ga)); out[expert*n_rows+row]=ga*sig*ua;}
    }
}

// MEGA down Q3K: 2D grid (n_rows, n_experts), direct pointer args, NO atomic
__global__ void mega8_q3k_down_kernel(
    const uint8_t* d0, const uint8_t* d1, const uint8_t* d2, const uint8_t* d3,
    const uint8_t* d4, const uint8_t* d5, const uint8_t* d6, const uint8_t* d7,
    const float* __restrict__ gate_buf, float* __restrict__ temp_out,
    int n_rows, int in_features, int inter_dim
) {
    int expert = blockIdx.y, row = blockIdx.x;
    const uint8_t* dW;
    switch(expert) {
        case 0: dW=d0; break; case 1: dW=d1; break; case 2: dW=d2; break; case 3: dW=d3; break;
        case 4: dW=d4; break; case 5: dW=d5; break; case 6: dW=d6; break; default: dW=d7; break;
    }
    const float* h = gate_buf + expert * inter_dim;
    int bpr=(in_features+255)/256;
    const uint8_t* rd=dW+(long long)row*bpr*110;
    float acc=0.0f;
    int tq=in_features/4;
    for(int qi=threadIdx.x;qi<tq;qi+=blockDim.x){
        int e=qi*4,blk=e/256,pos=e%256,sub=pos/16;
        const uint8_t* b=rd+blk*110;
        float d=__half2float(*(const __half*)(b+108));
        float sc[16]; q3k_unpack_scales(b+96,sc);
        float dl=d*sc[sub];
        uint8_t qb=b[32+pos/4];
        int q0=(qb&3),q1=((qb>>2)&3),q2=((qb>>4)&3),q3=((qb>>6)&3);
        int h0=(b[pos/8]>>(pos%8))&1,h1=(b[(pos+1)/8]>>((pos+1)%8))&1;
        int h2=(b[(pos+2)/8]>>((pos+2)%8))&1,h3=(b[(pos+3)/8]>>((pos+3)%8))&1;
        acc+=dl*(float)(q0-4+h0*4)*h[e]+dl*(float)(q1-4+h1*4)*h[e+1]
            +dl*(float)(q2-4+h2*4)*h[e+2]+dl*(float)(q3-4+h3*4)*h[e+3];
    }
    __shared__ float ws[32];
    acc=block_reduce(acc,ws);
    if(threadIdx.x==0) temp_out[expert * n_rows + row] = acc;
}

// MEGA down Q2K: 2D grid (n_rows, n_experts), direct pointer args, NO atomic
__global__ void mega8_q2k_down_kernel(
    const uint8_t* d0, const uint8_t* d1, const uint8_t* d2, const uint8_t* d3,
    const uint8_t* d4, const uint8_t* d5, const uint8_t* d6, const uint8_t* d7,
    const float* __restrict__ gate_buf, float* __restrict__ temp_out,
    int n_rows, int in_features, int inter_dim
) {
    int expert = blockIdx.y, row = blockIdx.x;
    const uint8_t* dW;
    switch(expert) {
        case 0: dW=d0; break; case 1: dW=d1; break; case 2: dW=d2; break; case 3: dW=d3; break;
        case 4: dW=d4; break; case 5: dW=d5; break; case 6: dW=d6; break; default: dW=d7; break;
    }
    const float* h = gate_buf + expert * inter_dim;
    int bpr=(in_features+255)/256;
    const uint8_t* rd=dW+(long long)row*bpr*84;
    float acc=0.0f;
    int tq=in_features/4;
    for(int qi=threadIdx.x;qi<tq;qi+=blockDim.x){
        int e=qi*4,blk=e/256,pos=e%256,sub=pos/16;
        const uint8_t* b=rd+blk*84;
        float d=__half2float(*(const __half*)(b+80)),dm=__half2float(*(const __half*)(b+82));
        float dl=d*(float)(b[sub]&0x0F),ml=dm*(float)(b[sub]>>4);
        uint8_t qb=b[16+pos/4];
        acc+=(dl*(float)(qb&3)-ml)*h[e]+(dl*(float)((qb>>2)&3)-ml)*h[e+1]
            +(dl*(float)((qb>>4)&3)-ml)*h[e+2]+(dl*(float)((qb>>6)&3)-ml)*h[e+3];
    }
    __shared__ float ws[32];
    acc=block_reduce(acc,ws);
    if(threadIdx.x==0) temp_out[expert * n_rows + row] = acc;
}

// Reduce expert outputs: weights as direct args (ZERO device copies)
__global__ void reduce8_expert_outputs_kernel(
    const float* __restrict__ temp_out, float* __restrict__ expert_out,
    int n_rows, int n_experts,
    float w0, float w1, float w2, float w3, float w4, float w5, float w6, float w7
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float ws[] = {w0,w1,w2,w3,w4,w5,w6,w7};
    float s = 0.0f;
    for (int k = 0; k < n_experts; k++)
        s += ws[k] * temp_out[k * n_rows + row];
    expert_out[row] += s;
}

// Fused reduce + accumulate: compute weighted sum from mega_temp and ADD to x directly.
// Eliminates: zero_kernel + reduce8_expert_outputs_kernel + weighted_add_kernel (3→1 launches)
__global__ void reduce8_and_accumulate_kernel(
    const float* __restrict__ temp_out, float* __restrict__ x,
    int n_rows, int n_experts,
    float w0, float w1, float w2, float w3, float w4, float w5, float w6, float w7
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float ws[] = {w0,w1,w2,w3,w4,w5,w6,w7};
    float s = 0.0f;
    for (int k = 0; k < n_experts; k++)
        s += ws[k] * temp_out[k * n_rows + row];
    x[row] += s;
}

// ============================================================
// V2 MEGA KERNELS — one 16-element sub-block per thread
//
// The v1 kernels walk 4 elements per thread per iteration, re-loading the block
// scale/d/dmin every time (~72 scalar 1-2 byte loads per thread). The Q3_K one
// also builds a float[16] scale array with a dynamic index, which the compiler
// must spill to LOCAL memory = extra global traffic on every iteration.
//
// v2 gives each thread exactly one 16-element sub-block, which is precisely:
//   - one uint32 of Q2_K qs (4 bytes)      - one scale byte
//   - one 32-bit d+dmin pair               - four float4 loads of x
// -> ~10 loads per thread instead of ~72, no local memory, wider transactions.
// ============================================================

// Q2_K gate+up+SiLU, vectorized. Requires 4-byte aligned weight pointers
// (guaranteed by cudaMalloc/torch: row stride 84*bpr and all offsets are %4==0).
__global__ void mega8_q2k_gus_v2_kernel(
    const uint8_t* g0, const uint8_t* g1, const uint8_t* g2, const uint8_t* g3,
    const uint8_t* g4, const uint8_t* g5, const uint8_t* g6, const uint8_t* g7,
    const uint8_t* u0, const uint8_t* u1, const uint8_t* u2, const uint8_t* u3,
    const uint8_t* u4, const uint8_t* u5, const uint8_t* u6, const uint8_t* u7,
    const float* __restrict__ x, float* __restrict__ out,
    int n_rows, int in_features
) {
    int expert = blockIdx.y, row = blockIdx.x;
    const uint8_t *gW, *uW;
    switch(expert) {
        case 0: gW=g0; uW=u0; break; case 1: gW=g1; uW=u1; break;
        case 2: gW=g2; uW=u2; break; case 3: gW=g3; uW=u3; break;
        case 4: gW=g4; uW=u4; break; case 5: gW=g5; uW=u5; break;
        case 6: gW=g6; uW=u6; break; default: gW=g7; uW=u7; break;
    }
    int bpr = (in_features + 255) / 256;
    long long byte_row = (long long)bpr * 84;
    const uint8_t* gr = gW + (long long)row * byte_row;
    const uint8_t* ur = uW + (long long)row * byte_row;
    int n_sub = in_features >> 4;
    float ga = 0.0f, ua = 0.0f;
    for (int s = threadIdx.x; s < n_sub; s += blockDim.x) {
        int blk = s >> 4, sub = s & 15;
        const uint8_t* gb = gr + blk * 84;
        const uint8_t* ub = ur + blk * 84;
        // d (bytes 80-81) and dmin (bytes 82-83) in ONE 32-bit load
        uint32_t gdd = *(const uint32_t*)(gb + 80);
        uint32_t udd = *(const uint32_t*)(ub + 80);
        float gd = __half2float(__ushort_as_half((unsigned short)(gdd & 0xFFFFu)));
        float gm = __half2float(__ushort_as_half((unsigned short)(gdd >> 16)));
        float ud = __half2float(__ushort_as_half((unsigned short)(udd & 0xFFFFu)));
        float um = __half2float(__ushort_as_half((unsigned short)(udd >> 16)));
        uint32_t gsm = gb[sub], usm = ub[sub];
        float gdl = gd * (float)(gsm & 0x0Fu), gml = gm * (float)(gsm >> 4);
        float udl = ud * (float)(usm & 0x0Fu), uml = um * (float)(usm >> 4);
        // 16 elements of qs in ONE 32-bit load each
        uint32_t gq = *(const uint32_t*)(gb + 16 + (sub << 2));
        uint32_t uq = *(const uint32_t*)(ub + 16 + (sub << 2));
        const float4* xv = (const float4*)(x + (blk << 8) + (sub << 4));
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float4 xx = xv[j];
            uint32_t gs = gq >> (j << 3), us = uq >> (j << 3);
            ga += (gdl*(float)(gs&3u)-gml)*xx.x + (gdl*(float)((gs>>2)&3u)-gml)*xx.y
                + (gdl*(float)((gs>>4)&3u)-gml)*xx.z + (gdl*(float)((gs>>6)&3u)-gml)*xx.w;
            ua += (udl*(float)(us&3u)-uml)*xx.x + (udl*(float)((us>>2)&3u)-uml)*xx.y
                + (udl*(float)((us>>4)&3u)-uml)*xx.z + (udl*(float)((us>>6)&3u)-uml)*xx.w;
        }
    }
    for (int off=16;off>0;off/=2){ga+=__shfl_down_sync(0xffffffff,ga,off);ua+=__shfl_down_sync(0xffffffff,ua,off);}
    __shared__ float ws[64];
    int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    if(lane==0){ws[wid]=ga;ws[32+wid]=ua;}
    __syncthreads();
    if(wid==0){
        int nw=(blockDim.x+31)>>5;
        ga=(lane<nw)?ws[lane]:0.0f; ua=(lane<nw)?ws[32+lane]:0.0f;
        for(int off=16;off>0;off/=2){ga+=__shfl_down_sync(0xffffffff,ga,off);ua+=__shfl_down_sync(0xffffffff,ua,off);}
        if(lane==0){float sig=1.0f/(1.0f+expf(-ga)); out[expert*n_rows+row]=ga*sig*ua;}
    }
}

// Q3_K down, vectorized. Block is 110 bytes so it is only 2-byte aligned —
// qs is read as 4 single bytes, but d/scale/hmask redundancy and the local-memory
// scale array are both gone (hmask: 2 loads per 16 elements instead of 16).
__global__ void mega8_q3k_down_v2_kernel(
    const uint8_t* d0, const uint8_t* d1, const uint8_t* d2, const uint8_t* d3,
    const uint8_t* d4, const uint8_t* d5, const uint8_t* d6, const uint8_t* d7,
    const float* __restrict__ gate_buf, float* __restrict__ temp_out,
    int n_rows, int in_features, int inter_dim
) {
    int expert = blockIdx.y, row = blockIdx.x;
    const uint8_t* dW;
    switch(expert) {
        case 0: dW=d0; break; case 1: dW=d1; break; case 2: dW=d2; break; case 3: dW=d3; break;
        case 4: dW=d4; break; case 5: dW=d5; break; case 6: dW=d6; break; default: dW=d7; break;
    }
    const float* h = gate_buf + (long long)expert * inter_dim;
    int bpr = (in_features + 255) / 256;
    const uint8_t* rd = dW + (long long)row * bpr * 110;
    int n_sub = in_features >> 4;
    float acc = 0.0f;
    for (int s = threadIdx.x; s < n_sub; s += blockDim.x) {
        int blk = s >> 4, sub = s & 15;
        const uint8_t* b = rd + blk * 110;
        float dl = __half2float(*(const __half*)(b + 108)) * q3k_scale_at(b + 96, sub);
        const uint8_t* qp = b + 32 + (sub << 2);
        // hmask bits for these 16 elements live in exactly 2 bytes
        uint32_t hm = (uint32_t)b[sub << 1] | ((uint32_t)b[(sub << 1) + 1] << 8);
        const float4* hv = (const float4*)(h + (blk << 8) + (sub << 4));
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t qb = qp[j];
            float4 hh = hv[j];
            int base = j << 2;
            acc += dl*(float)((int)( qb     &3u)-4+(int)((hm>>(base  ))&1u)*4)*hh.x
                 + dl*(float)((int)((qb>>2)&3u)-4+(int)((hm>>(base+1))&1u)*4)*hh.y
                 + dl*(float)((int)((qb>>4)&3u)-4+(int)((hm>>(base+2))&1u)*4)*hh.z
                 + dl*(float)((int)((qb>>6)&3u)-4+(int)((hm>>(base+3))&1u)*4)*hh.w;
        }
    }
    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) temp_out[expert * n_rows + row] = acc;
}

// Q2_K down, vectorized (used when the down projection is Q2_K instead of Q3_K).
__global__ void mega8_q2k_down_v2_kernel(
    const uint8_t* d0, const uint8_t* d1, const uint8_t* d2, const uint8_t* d3,
    const uint8_t* d4, const uint8_t* d5, const uint8_t* d6, const uint8_t* d7,
    const float* __restrict__ gate_buf, float* __restrict__ temp_out,
    int n_rows, int in_features, int inter_dim
) {
    int expert = blockIdx.y, row = blockIdx.x;
    const uint8_t* dW;
    switch(expert) {
        case 0: dW=d0; break; case 1: dW=d1; break; case 2: dW=d2; break; case 3: dW=d3; break;
        case 4: dW=d4; break; case 5: dW=d5; break; case 6: dW=d6; break; default: dW=d7; break;
    }
    const float* h = gate_buf + (long long)expert * inter_dim;
    int bpr = (in_features + 255) / 256;
    const uint8_t* rd = dW + (long long)row * bpr * 84;
    int n_sub = in_features >> 4;
    float acc = 0.0f;
    for (int s = threadIdx.x; s < n_sub; s += blockDim.x) {
        int blk = s >> 4, sub = s & 15;
        const uint8_t* b = rd + blk * 84;
        uint32_t dd = *(const uint32_t*)(b + 80);
        float d = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFFu)));
        float dm = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
        uint32_t sm = b[sub];
        float dl = d * (float)(sm & 0x0Fu), ml = dm * (float)(sm >> 4);
        uint32_t q = *(const uint32_t*)(b + 16 + (sub << 2));
        const float4* hv = (const float4*)(h + (blk << 8) + (sub << 4));
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float4 hh = hv[j];
            uint32_t qs = q >> (j << 3);
            acc += (dl*(float)(qs&3u)-ml)*hh.x + (dl*(float)((qs>>2)&3u)-ml)*hh.y
                 + (dl*(float)((qs>>4)&3u)-ml)*hh.z + (dl*(float)((qs>>6)&3u)-ml)*hh.w;
        }
    }
    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) temp_out[expert * n_rows + row] = acc;
}

// Q2K down that ADDS to output (for shared expert → accumulate directly into x)
__global__ void q2k_down_accumulate_kernel(
    const uint8_t* __restrict__ W_data,
    const float*   __restrict__ h,
    float*         __restrict__ x,
    int n_rows, int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int blocks_per_row = (in_features + 255) / 256;
    int bytes_per_row = blocks_per_row * 84;
    const uint8_t* row_data = W_data + (long long)row * bytes_per_row;
    float acc = 0.0f;
    int total_qs = in_features / 4;
    for (int qi = threadIdx.x; qi < total_qs; qi += blockDim.x) {
        int elem = qi * 4;
        int blk = elem / 256;
        int pos = elem % 256;
        int sub = pos / 16;
        const uint8_t* block = row_data + blk * 84;
        float d   = __half2float(*(const __half*)(block + 80));
        float dm  = __half2float(*(const __half*)(block + 82));
        uint8_t sc = block[sub];
        float dl = d  * (float)(sc & 0x0F);
        float ml = dm * (float)(sc >> 4);
        uint8_t qb = block[16 + pos / 4];
        float v0 = dl * (float)( qb       & 3) - ml;
        float v1 = dl * (float)((qb >> 2) & 3) - ml;
        float v2 = dl * (float)((qb >> 4) & 3) - ml;
        float v3 = dl * (float)((qb >> 6) & 3) - ml;
        acc += v0*h[elem] + v1*h[elem+1] + v2*h[elem+2] + v3*h[elem+3];
    }
    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) x[row] += acc;
}

// Q3K down that ADDS to output
__global__ void q3k_down_accumulate_kernel(
    const uint8_t* __restrict__ W_data,
    const float*   __restrict__ h,
    float*         __restrict__ x,
    int n_rows, int in_features
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int blocks_per_row = (in_features + 255) / 256;
    int bytes_per_row = blocks_per_row * 110;
    const uint8_t* row_data = W_data + (long long)row * bytes_per_row;
    float acc = 0.0f;
    int total_qs = in_features / 4;
    for (int qi = threadIdx.x; qi < total_qs; qi += blockDim.x) {
        int elem = qi * 4;
        int blk = elem / 256;
        int pos = elem % 256;
        int sub = pos / 16;
        const uint8_t* block = row_data + blk * 110;
        float d = __half2float(*(const __half*)(block + 108));
        float scales[16];
        q3k_unpack_scales(block + 96, scales);
        float dl = d * scales[sub];
        uint8_t qb = block[32 + pos / 4];
        int q0 = ( qb       & 3), q1 = ((qb >> 2) & 3);
        int q2 = ((qb >> 4) & 3), q3 = ((qb >> 6) & 3);
        int h0 = (block[pos / 8] >> (pos % 8)) & 1;
        int h1 = (block[(pos+1) / 8] >> ((pos+1) % 8)) & 1;
        int h2 = (block[(pos+2) / 8] >> ((pos+2) % 8)) & 1;
        int h3 = (block[(pos+3) / 8] >> ((pos+3) % 8)) & 1;
        float v0 = dl * (float)(q0 - 4 + h0 * 4);
        float v1 = dl * (float)(q1 - 4 + h1 * 4);
        float v2 = dl * (float)(q2 - 4 + h2 * 4);
        float v3 = dl * (float)(q3 - 4 + h3 * 4);
        acc += v0*h[elem] + v1*h[elem+1] + v2*h[elem+2] + v3*h[elem+3];
    }
    __shared__ float ws[32];
    acc = block_reduce(acc, ws);
    if (threadIdx.x == 0) x[row] += acc;
}

static void* d_mega_ptrs = NULL;
static float* d_mega_wts = NULL;
static int d_mega_cap = 0;
static void ensure_mega_bufs(int n) {
    if (n <= d_mega_cap) return;
    if (d_mega_ptrs) cudaFree(d_mega_ptrs);
    if (d_mega_wts) cudaFree(d_mega_wts);
    cudaMalloc(&d_mega_ptrs, 3 * n * sizeof(void*));
    cudaMalloc(&d_mega_wts, n * sizeof(float));
    d_mega_cap = n;
}

// Entire MoE: shared + MEGA8 experts (direct pointer args, ZERO cudaMemcpy) + residual
__declspec(dllexport) void fused_moe_experts(
    const float* norm_buf, int hidden_dim,
    const uint8_t* sh_gate, const uint8_t* sh_up, const uint8_t* sh_down,
    int sh_inter, int sh_in, int sh_down_rows, int sh_down_in, int sh_gqt, int sh_dqt,
    const void** eg, const void** eu, const void** ed,
    const float* weights, int n_exp,
    int e_inter, int e_in, int e_down_rows, int e_down_in, int e_gqt, int e_dqt,
    float* gate_buf, float* expert_out, float* mega_temp, float* x
) {
    int T = 256;
    // Shared expert: gate_up_silu → down → accumulate directly into x
    if (sh_gate) {
        if (sh_gqt == 0)
            q2k_gate_up_silu_kernel<<<sh_inter, T>>>(sh_gate, sh_up, norm_buf, gate_buf, sh_inter, sh_in);
        else {
            q3k_matvec_kernel<<<sh_inter, T>>>(sh_gate, norm_buf, gate_buf, sh_inter, sh_in);
            q3k_matvec_kernel<<<sh_inter, T>>>(sh_up, norm_buf, mega_temp, sh_inter, sh_in);
            swiglu_fuse_kernel<<<(sh_inter+T-1)/T, T>>>(gate_buf, mega_temp, sh_inter);
        }
        if (sh_dqt == 0)
            q2k_down_accumulate_kernel<<<sh_down_rows, T>>>(sh_down, gate_buf, x, sh_down_rows, sh_down_in);
        else
            q3k_down_accumulate_kernel<<<sh_down_rows, T>>>(sh_down, gate_buf, x, sh_down_rows, sh_down_in);
    }
    // Routed experts: mega gate_up_silu → mega down → reduce+accumulate into x
    if (n_exp > 0) {
        const uint8_t* g[8]={0,0,0,0,0,0,0,0}, *u[8]={0,0,0,0,0,0,0,0}, *d[8]={0,0,0,0,0,0,0,0};
        float w[8]={0,0,0,0,0,0,0,0};
        for(int i=0;i<n_exp&&i<8;i++){g[i]=(const uint8_t*)eg[i];u[i]=(const uint8_t*)eu[i];d[i]=(const uint8_t*)ed[i];w[i]=weights[i];}
        dim3 gus_grid(e_inter, n_exp);
        if (e_gqt == 0)
            mega8_q2k_gate_up_silu_kernel<<<gus_grid, T>>>(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],
                u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7], norm_buf, gate_buf, e_inter, e_in);
        else {
            for(int i=0;i<n_exp;i++){
                q3k_matvec_kernel<<<e_inter,T>>>(g[i],norm_buf,gate_buf+i*e_inter,e_inter,e_in);
                q3k_matvec_kernel<<<e_inter,T>>>(u[i],norm_buf,mega_temp,e_inter,e_in);
                swiglu_fuse_kernel<<<(e_inter+T-1)/T,T>>>(gate_buf+i*e_inter,mega_temp,e_inter);
            }
        }
        dim3 dn_grid(e_down_rows, n_exp);
        if (e_dqt == 0)
            mega8_q2k_down_kernel<<<dn_grid, T>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                gate_buf, mega_temp, e_down_rows, e_down_in, e_inter);
        else
            mega8_q3k_down_kernel<<<dn_grid, T>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                gate_buf, mega_temp, e_down_rows, e_down_in, e_inter);
        reduce8_and_accumulate_kernel<<<(e_down_rows+T-1)/T, T>>>(mega_temp, x,
            e_down_rows, n_exp, w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    }
}

// Entire attention layer in ONE C call
__declspec(dllexport) void fused_attention_layer(
    float* x, int hidden_dim, const float* anorm,
    const uint8_t* qaW, int qaR, int qaI, int qaQt,
    const float* qaNorm, int qaNormN,
    const uint8_t* qbW, int qbR, int qbI, int qbQt,
    const uint8_t* kvaW, int kvaR, int kvaI, int kvaQt,
    const float* kvaNorm, int kvaNormN,
    const uint8_t* outW, int outR, int outI, int outQt,
    float* normB, float* qaB, float* qbB, float* kvaB, float* attnB
) {
    int T = 256;
    rms_norm_kernel<<<1, T>>>(x, anorm, normB, hidden_dim);
    if (qaQt==0) q2k_matvec_kernel<<<qaR, T>>>(qaW, normB, qaB, qaR, qaI);
    else q3k_matvec_kernel<<<qaR, T>>>(qaW, normB, qaB, qaR, qaI);
    if (qaNorm) rms_norm_kernel<<<1, T>>>(qaB, qaNorm, qaB, qaNormN);
    if (qbW) {
        if (qbQt==0) q2k_matvec_kernel<<<qbR, T>>>(qbW, qaB, qbB, qbR, qbI);
        else q3k_matvec_kernel<<<qbR, T>>>(qbW, qaB, qbB, qbR, qbI);
    }
    if (kvaW) {
        if (kvaQt==0) q2k_matvec_kernel<<<kvaR, T>>>(kvaW, normB, kvaB, kvaR, kvaI);
        else q3k_matvec_kernel<<<kvaR, T>>>(kvaW, normB, kvaB, kvaR, kvaI);
    }
    if (kvaNorm) rms_norm_kernel<<<1, T>>>(kvaB, kvaNorm, kvaB, kvaNormN);
    const float* oi = qbW ? qbB : qaB;
    if (outQt==0) q2k_matvec_kernel<<<outR, T>>>(outW, oi, attnB, outR, outI);
    else q3k_matvec_kernel<<<outR, T>>>(outW, oi, attnB, outR, outI);
    weighted_add_kernel<<<(hidden_dim+T-1)/T, T>>>(x, attnB, 1.0f, hidden_dim);
}

// Bisect switches: let the host null out one contributor at a time to find which
// component is inflating the residual stream. g_no_shared MUST also bypass the
// preamble CUDA graph, because a replayed graph is baked and ignores host flags.
static int g_no_shared = 0;
static int g_no_attn = 0;
__declspec(dllexport) void set_bisect(int no_shared, int no_attn) {
    g_no_shared = no_shared ? 1 : 0;
    g_no_attn = no_attn ? 1 : 0;
}

// Multiplier on the attention score scale (1.0 = textbook 1/sqrt(head_dim)).
static float g_score_scale = 1.0f;
__declspec(dllexport) void set_score_scale(float s) { g_score_scale = s; }
__declspec(dllexport) float get_score_scale() { return g_score_scale; }

// ============================================================
// MLA — Multi-head Latent Attention with a compressed KV cache
//
// The old fused_attention_layer computed kv_a and THREW IT AWAY, then applied the
// output projection straight to the query. No scores, no softmax, no history. This
// is the real thing.
//
// Shapes for glm-dsa (read from the GGUF, not assumed):
//   n_heads=64  qk_nope=192  qk_rope=64  kv_lora=512  v_head=256
//   q_b out = 64*(192+64) = 16384 ; kv_a out = 512+64 = 576 (k_rope shared = MQA)
//   k_b[h]: 192 -> 512 (absorbs q_nope INTO compressed space)
//   v_b[h]: 512 -> 256
//
// Absorption is what makes this cheap: instead of decompressing k for every cached
// position, push the query through k_b once and score directly against the stored
// 512-d latent. Cache cost is 576 floats/token for ALL 64 heads.
// ============================================================

// Rotary embedding. Two conventions exist and they are NOT interchangeable:
//   style 0 NeoX  — element i pairs with i + dim/2
//   style 1 GPT-J — element 2i pairs with 2i+1 (interleaved)
// At pos=0 both are the identity, so only pos>0 can tell them apart.
// `stride` lets this walk q_rope slices embedded inside a larger per-head vector
// (q is laid out [head][qk_nope | qk_rope], so stride = qk_nope + qk_rope).
// Default 1 (GPT-J interleaved): llama.cpp maps LLM_ARCH_GLM_DSA to
// LLAMA_ROPE_TYPE_NORM (llama-model.cpp rope-type switch), which rotates
// adjacent pairs (2i, 2i+1) — NOT NeoX half-split.
static int g_rope_style = 1;
__declspec(dllexport) void set_rope_style(int s) { g_rope_style = s ? 1 : 0; }
__declspec(dllexport) int  get_rope_style() { return g_rope_style; }

__global__ void rope_neox_kernel(float* __restrict__ v, int n_vec, int dim, int stride,
                                 int pos, float base, int style) {
    int vec = blockIdx.x;
    if (vec >= n_vec) return;
    int half = dim >> 1;
    float* p = v + (long long)vec * stride;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float inv = __powf(base, -((float)(2 * i) / (float)dim));
        float th = (float)pos * inv;
        float c = __cosf(th), s = __sinf(th);
        int ia = style ? (2 * i)     : i;
        int ib = style ? (2 * i + 1) : (i + half);
        float a = p[ia], b = p[ib];
        p[ia] = a * c - b * s;
        p[ib] = a * s + b * c;
    }
}

// qc[h] = k_b[h] @ q_nope[h]   (in 192 -> out 512), k_b is fp16 [h][out][in]
__global__ void mla_absorb_q_kernel(const __half* __restrict__ kb,
                                    const float* __restrict__ qb, float* __restrict__ qc,
                                    int n_heads, int qk_nope, int qk_rope, int kv_lora) {
    int h = blockIdx.y, o = blockIdx.x;
    if (h >= n_heads || o >= kv_lora) return;
    const __half* W = kb + (long long)h * kv_lora * qk_nope + (long long)o * qk_nope;
    const float* q = qb + (long long)h * (qk_nope + qk_rope);
    float s = 0.0f;
    for (int i = threadIdx.x; i < qk_nope; i += blockDim.x)
        s += __half2float(W[i]) * q[i];
    __shared__ float ws[32];
    s = block_reduce(s, ws);
    if (threadIdx.x == 0) qc[(long long)h * kv_lora + o] = s;
}

// score[h][t] = (qc[h].c[t] + q_rope[h].krope[t]) * scale, for t in [0, pos]
__global__ void mla_scores_kernel(const float* __restrict__ qc, const float* __restrict__ qb,
                                  const float* __restrict__ c_cache,
                                  const float* __restrict__ kr_cache,
                                  float* __restrict__ scores,
                                  int n_heads, int qk_nope, int qk_rope, int kv_lora,
                                  int n_ctx, int max_seq, float scale) {
    int h = blockIdx.y, t = blockIdx.x;
    if (h >= n_heads || t >= n_ctx) return;
    const float* qcv = qc + (long long)h * kv_lora;
    const float* qr  = qb + (long long)h * (qk_nope + qk_rope) + qk_nope;
    const float* cv  = c_cache  + (long long)t * kv_lora;
    const float* kr  = kr_cache + (long long)t * qk_rope;
    float s = 0.0f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) s += qcv[i] * cv[i];
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) s += qr[i] * kr[i];
    __shared__ float ws[32];
    s = block_reduce(s, ws);
    if (threadIdx.x == 0) scores[(long long)h * max_seq + t] = s * scale;
}

// Softmax across the n_ctx cached positions, one block per head.
__global__ void mla_softmax_kernel(float* __restrict__ scores, int n_ctx, int max_seq) {
    int h = blockIdx.x;
    float* row = scores + (long long)h * max_seq;
    __shared__ float sm[32];
    float m = -1e30f;
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) m = fmaxf(m, row[t]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = -1e30f;
        int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; i++) mm = fmaxf(mm, sm[i]);
        sm[31] = mm;
    }
    __syncthreads();
    float mx = sm[31];
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) {
        float e = __expf(row[t] - mx);
        row[t] = e; sum += e;
    }
    __shared__ float ws[32];
    sum = block_reduce(sum, ws);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = 1.0f / (sum + 1e-9f);
    __syncthreads();
    for (int t = threadIdx.x; t < n_ctx; t += blockDim.x) row[t] *= inv;
}

// oc[h] = sum_t attn[h][t] * c[t]   — weighted sum IN COMPRESSED SPACE (v_b is linear,
// so decompressing after the sum is identical to decompressing every position first).
__global__ void mla_weighted_c_kernel(const float* __restrict__ scores,
                                      const float* __restrict__ c_cache,
                                      float* __restrict__ oc,
                                      int n_heads, int kv_lora, int n_ctx, int max_seq) {
    int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads || d >= kv_lora) return;
    const float* row = scores + (long long)h * max_seq;
    float s = 0.0f;
    for (int t = 0; t < n_ctx; t++) s += row[t] * c_cache[(long long)t * kv_lora + d];
    oc[(long long)h * kv_lora + d] = s;
}

// head[h] = v_b[h] @ oc[h]   (in 512 -> out 256), v_b fp16 [h][out][in]
__global__ void mla_v_kernel(const __half* __restrict__ vb, const float* __restrict__ oc,
                             float* __restrict__ heads,
                             int n_heads, int kv_lora, int v_head) {
    int h = blockIdx.y, o = blockIdx.x;
    if (h >= n_heads || o >= v_head) return;
    const __half* W = vb + (long long)h * v_head * kv_lora + (long long)o * kv_lora;
    const float* v = oc + (long long)h * kv_lora;
    float s = 0.0f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x)
        s += __half2float(W[i]) * v[i];
    __shared__ float ws[32];
    s = block_reduce(s, ws);
    if (threadIdx.x == 0) heads[(long long)h * v_head + o] = s;
}

// Copy this step's latent + roped key into the cache slot for `pos`.
__global__ void mla_cache_store_kernel(const float* __restrict__ kva,
                                       float* __restrict__ c_cache, float* __restrict__ kr_cache,
                                       int pos, int kv_lora, int qk_rope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_lora) c_cache[(long long)pos * kv_lora + i] = kva[i];
    else if (i < kv_lora + qk_rope)
        kr_cache[(long long)pos * qk_rope + (i - kv_lora)] = kva[i];
}

__declspec(dllexport) void mla_attention(
    float* x, int hidden_dim, const float* anorm,
    const uint8_t* qaW, int qaR, int qaI, int qaQt, const float* qaNorm,
    const uint8_t* qbW, int qbR, int qbI, int qbQt,
    const uint8_t* kvaW, int kvaR, int kvaI, int kvaQt, const float* kvaNorm,
    const __half* kb, const __half* vb,
    const uint8_t* outW, int outR, int outI, int outQt,
    float* c_cache, float* kr_cache, int pos, int max_seq,
    float* normB, float* qaB, float* qbB, float* kvaB,
    float* qcB, float* scoreB, float* ocB, float* headB, float* attnB,
    int n_heads, int qk_nope, int qk_rope, int kv_lora, int v_head, float rope_base
) {
    int T = 256;
    int n_ctx = pos + 1;

    // --- 1. norm + query path -> [n_heads, qk_nope+qk_rope]
    rms_norm_kernel<<<1, T>>>(x, anorm, normB, hidden_dim);
    if (qaQt == 0) q2k_matvec_kernel<<<qaR, T>>>(qaW, normB, qaB, qaR, qaI);
    else           q3k_matvec_kernel<<<qaR, T>>>(qaW, normB, qaB, qaR, qaI);
    if (qaNorm) rms_norm_kernel<<<1, T>>>(qaB, qaNorm, qaB, qaR);
    if (qbQt == 0) q2k_matvec_kernel<<<qbR, T>>>(qbW, qaB, qbB, qbR, qbI);
    else           q3k_matvec_kernel<<<qbR, T>>>(qbW, qaB, qbB, qbR, qbI);

    // --- 2. kv path -> [kv_lora | qk_rope]; norm applies to the latent only
    if (kvaQt == 0) q2k_matvec_kernel<<<kvaR, T>>>(kvaW, normB, kvaB, kvaR, kvaI);
    else            q3k_matvec_kernel<<<kvaR, T>>>(kvaW, normB, kvaB, kvaR, kvaI);
    if (kvaNorm) rms_norm_kernel<<<1, T>>>(kvaB, kvaNorm, kvaB, kv_lora);

    // --- 3. RoPE at this position. q_rope is the tail of each head's slice, so walk it
    //        with stride (qk_nope+qk_rope); k_rope is a single shared vector (MQA).
    rope_neox_kernel<<<n_heads, 64>>>(qbB + qk_nope, n_heads, qk_rope,
                                      qk_nope + qk_rope, pos, rope_base, g_rope_style);
    rope_neox_kernel<<<1, 64>>>(kvaB + kv_lora, 1, qk_rope, qk_rope, pos, rope_base,
                                g_rope_style);

    // --- 4. append to cache
    mla_cache_store_kernel<<<(kv_lora + qk_rope + T - 1) / T, T>>>(
        kvaB, c_cache, kr_cache, pos, kv_lora, qk_rope);

    // --- 5. absorb query, score against all cached latents, softmax
    dim3 ab(kv_lora, n_heads);
    mla_absorb_q_kernel<<<ab, 128>>>(kb, qbB, qcB, n_heads, qk_nope, qk_rope, kv_lora);
    dim3 sc(n_ctx, n_heads);
    // Measured: with the textbook 1/sqrt(qk_nope+qk_rope) the scaled score spread is
    // 0.03-0.64 and the softmax mean-pools (entropy 2.069 vs uniform 2.079). The
    // multiplier is runtime-settable so the right scale can be found empirically.
    float scale = g_score_scale / sqrtf((float)(qk_nope + qk_rope));
    mla_scores_kernel<<<sc, 128>>>(qcB, qbB, c_cache, kr_cache, scoreB,
                                   n_heads, qk_nope, qk_rope, kv_lora, n_ctx, max_seq, scale);
    mla_softmax_kernel<<<n_heads, 256>>>(scoreB, n_ctx, max_seq);

    // --- 6. weighted latent sum, decompress to v, project out, residual
    dim3 wc((kv_lora + 127) / 128, n_heads);
    mla_weighted_c_kernel<<<wc, 128>>>(scoreB, c_cache, ocB, n_heads, kv_lora, n_ctx, max_seq);
    dim3 vg(v_head, n_heads);
    mla_v_kernel<<<vg, 128>>>(vb, ocB, headB, n_heads, kv_lora, v_head);
    if (outQt == 0) q2k_matvec_kernel<<<outR, T>>>(outW, headB, attnB, outR, outI);
    else            q3k_matvec_kernel<<<outR, T>>>(outW, headB, attnB, outR, outI);
    // Bisect: still fill the KV cache, but contribute nothing to the residual.
    if (!g_no_attn)
        weighted_add_kernel<<<(hidden_dim + T - 1) / T, T>>>(x, attnB, 1.0f, hidden_dim);
}

__declspec(dllexport) void cuda_rms_norm(const float* x, const float* w, float* out, int n) {
    rms_norm_kernel<<<1, 256>>>(x, w, out, n);
}

__declspec(dllexport) void cuda_weighted_add(float* y, const float* src, float weight, int n) {
    weighted_add_kernel<<<(n + 255) / 256, 256>>>(y, src, weight, n);
}

__declspec(dllexport) void cuda_zero(float* buf, int n) {
    zero_kernel<<<(n + 255) / 256, 256>>>(buf, n);
}

__declspec(dllexport) void moe_router_topk(
    const __half* router_W,
    const float* bias,
    const float* x,
    float* router_buf,
    float* topk_vals_buf,
    int* topk_ids_buf,
    int n_experts, int hidden_dim, int top_k,
    int* host_ids,
    float* host_vals
) {
    fp16_matvec_kernel<<<n_experts, 256>>>(router_W, x, router_buf, n_experts, hidden_dim);
    topk_kernel<<<1, 1>>>(router_buf, bias, topk_vals_buf, topk_ids_buf, n_experts, top_k);
    cudaDeviceSynchronize();
    cudaMemcpy(host_ids, topk_ids_buf, top_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vals, topk_vals_buf, top_k * sizeof(float), cudaMemcpyDeviceToHost);
}

__declspec(dllexport) void sync_device() {
    cudaDeviceSynchronize();
}

// CUDA Graph capture/replay — eliminates per-kernel launch overhead
// Slots 0-77: attention graphs. Slots 78-155: MoE preamble graphs (layer 3-77 → slot 78+layer-3).
static cudaGraphExec_t g_graph_execs[256] = {0};

__declspec(dllexport) void begin_graph_capture() {
    cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed);
}

__declspec(dllexport) int end_graph_capture(int slot) {
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(0, &graph);
    if (err != cudaSuccess) return -1;
    if (g_graph_execs[slot]) {
        cudaGraphExecDestroy(g_graph_execs[slot]);
        g_graph_execs[slot] = 0;
    }
    err = cudaGraphInstantiate(&g_graph_execs[slot], graph, 0);
    cudaGraphDestroy(graph);
    return (err == cudaSuccess) ? 0 : -2;
}

__declspec(dllexport) void replay_graph(int slot) {
    if (g_graph_execs[slot])
        cudaGraphLaunch(g_graph_execs[slot], 0);
}

// ============================================================
// Full MoE layer in ONE C call — eliminates Python loop overhead
// ============================================================

// Host-side expert pointer table (populated once after cold run)
static const uint8_t* h_expert_gate[78][128];
static const uint8_t* h_expert_up[78][128];
static const uint8_t* h_expert_down[78][128];

__declspec(dllexport) void register_expert(
    int layer, int expert,
    const uint8_t* gate, const uint8_t* up, const uint8_t* down
) {
    if (layer < 78 && expert < 128) {
        h_expert_gate[layer][expert] = gate;
        h_expert_up[layer][expert] = up;
        h_expert_down[layer][expert] = down;
    }
}

// Per-layer MoE metadata (registered once, used every token)
struct MoeLayerInfo {
    const float* ffn_norm_w;
    const __half* router_W;
    const float* router_bias;
    float* router_buf;
    float* topk_vals_buf;
    int* topk_ids_buf;
    int n_experts_total, top_k, hidden_dim;
    const uint8_t* sh_gate;
    const uint8_t* sh_up;
    const uint8_t* sh_down;
    int sh_inter, sh_in, sh_dr, sh_di, sh_gqt, sh_dqt;
    int e_inter, e_in, e_dr, e_di, e_gqt, e_dqt;
    int valid;
};

static MoeLayerInfo moe_info[78] = {};

__declspec(dllexport) void register_moe_layer(
    int layer_idx,
    const float* ffn_norm_w,
    const __half* router_W, const float* router_bias,
    float* router_buf, float* topk_vals_buf, int* topk_ids_buf,
    int n_experts_total, int top_k, int hidden_dim,
    const uint8_t* sh_gate, const uint8_t* sh_up, const uint8_t* sh_down,
    int sh_inter, int sh_in, int sh_dr, int sh_di, int sh_gqt, int sh_dqt,
    int e_inter, int e_in, int e_dr, int e_di, int e_gqt, int e_dqt
) {
    if (layer_idx >= 78) return;
    MoeLayerInfo* m = &moe_info[layer_idx];
    m->ffn_norm_w = ffn_norm_w;
    m->router_W = router_W; m->router_bias = router_bias;
    m->router_buf = router_buf; m->topk_vals_buf = topk_vals_buf; m->topk_ids_buf = topk_ids_buf;
    m->n_experts_total = n_experts_total; m->top_k = top_k; m->hidden_dim = hidden_dim;
    m->sh_gate = sh_gate; m->sh_up = sh_up; m->sh_down = sh_down;
    m->sh_inter = sh_inter; m->sh_in = sh_in; m->sh_dr = sh_dr; m->sh_di = sh_di;
    m->sh_gqt = sh_gqt; m->sh_dqt = sh_dqt;
    m->e_inter = e_inter; m->e_in = e_in; m->e_dr = e_dr; m->e_di = e_di;
    m->e_gqt = e_gqt; m->e_dqt = e_dqt;
    m->valid = 1;
}

// Honesty counters: how many of the router's top-k picks were actually resident.
// If g_experts_requested != g_experts_executed, the layer silently did LESS work
// than the model specifies, and any tok/s number from that run is INFLATED.
static long long g_experts_requested = 0;
static long long g_experts_executed  = 0;

// Miss log: every (layer, expert) the router asked for that was NOT resident.
// Python drains this after a forward pass and pages those experts in, so the next
// token runs at full top-k instead of silently dropping them.
#define MISS_LOG_CAP 4096
static int g_miss_log[MISS_LOG_CAP][2];
static int g_miss_count = 0;
static long long g_miss_dropped = 0;   // overflow, so a full log is never mistaken for "no misses"

static inline void log_miss(int layer, int eid) {
    if (g_miss_count < MISS_LOG_CAP) {
        g_miss_log[g_miss_count][0] = layer;
        g_miss_log[g_miss_count][1] = eid;
        g_miss_count++;
    } else {
        g_miss_dropped++;
    }
}

// Route log: EVERY (layer, expert) the router picks, resident or not. This is the
// ground truth for "which experts does this text actually use" — reading the shared
// topk buffer from Python after the forward only ever shows the LAST layer.
#define ROUTE_LOG_CAP 32768
static int g_route_log[ROUTE_LOG_CAP][2];
static int g_route_count = 0;
static int g_route_on = 0;
static long long g_route_dropped = 0;

__declspec(dllexport) void set_route_logging(int on) { g_route_on = on ? 1 : 0; }

static inline void log_route(int layer, int eid) {
    if (!g_route_on) return;
    if (g_route_count < ROUTE_LOG_CAP) {
        g_route_log[g_route_count][0] = layer;
        g_route_log[g_route_count][1] = eid;
        g_route_count++;
    } else {
        g_route_dropped++;
    }
}

__declspec(dllexport) int drain_route_log(int* out, int max_pairs) {
    int n = (g_route_count < max_pairs) ? g_route_count : max_pairs;
    for (int i = 0; i < n; i++) { out[2*i] = g_route_log[i][0]; out[2*i+1] = g_route_log[i][1]; }
    g_route_count = 0;
    return n;
}
__declspec(dllexport) long long get_route_dropped() { return g_route_dropped; }

// Copies up to max_pairs (layer,expert) pairs into out[] and clears the log.
// Returns the number of pairs written.
__declspec(dllexport) int drain_miss_log(int* out, int max_pairs) {
    int n = (g_miss_count < max_pairs) ? g_miss_count : max_pairs;
    for (int i = 0; i < n; i++) { out[2*i] = g_miss_log[i][0]; out[2*i+1] = g_miss_log[i][1]; }
    g_miss_count = 0;
    return n;
}
__declspec(dllexport) long long get_miss_dropped() { return g_miss_dropped; }

__declspec(dllexport) void full_moe_layer(
    float* x, int layer_idx,
    float* norm_buf, float* gate_buf, float* mega_temp
) {
    if (layer_idx >= 78 || !moe_info[layer_idx].valid) return;
    MoeLayerInfo* m = &moe_info[layer_idx];
    int T = 256;

    // 1. RMS norm
    rms_norm_kernel<<<1, T>>>(x, m->ffn_norm_w, norm_buf, m->hidden_dim);

    // 2. Router (launches async, results needed after shared expert)
    fp16_matvec_kernel<<<m->n_experts_total, T>>>(m->router_W, norm_buf, m->router_buf, m->n_experts_total, m->hidden_dim);
    topk_kernel<<<1, 1>>>(m->router_buf, m->router_bias, m->topk_vals_buf, m->topk_ids_buf, m->n_experts_total, m->top_k);

    // 3. Shared expert (runs while router results queue up)
    if (m->sh_gate) {
        if (m->sh_gqt == 0)
            q2k_gate_up_silu_kernel<<<m->sh_inter, T>>>(m->sh_gate, m->sh_up, norm_buf, gate_buf, m->sh_inter, m->sh_in);
        else {
            q3k_matvec_kernel<<<m->sh_inter, T>>>(m->sh_gate, norm_buf, gate_buf, m->sh_inter, m->sh_in);
            q3k_matvec_kernel<<<m->sh_inter, T>>>(m->sh_up, norm_buf, mega_temp, m->sh_inter, m->sh_in);
            swiglu_fuse_kernel<<<(m->sh_inter+T-1)/T, T>>>(gate_buf, mega_temp, m->sh_inter);
        }
        if (m->sh_dqt == 0)
            q2k_down_accumulate_kernel<<<m->sh_dr, T>>>(m->sh_down, gate_buf, x, m->sh_dr, m->sh_di);
        else
            q3k_down_accumulate_kernel<<<m->sh_dr, T>>>(m->sh_down, gate_buf, x, m->sh_dr, m->sh_di);
    }

    // 4. Sync + read router results (shared expert work was "free" GPU time)
    cudaDeviceSynchronize();
    int h_ids[8]; float h_vals[8];
    cudaMemcpy(h_ids, m->topk_ids_buf, m->top_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals, m->topk_vals_buf, m->top_k * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Look up expert pointers from host table (C-side, zero Python overhead)
    const uint8_t* g[8]={0}, *u[8]={0}, *d[8]={0}; float w[8]={0};
    int n_valid = 0;
    for (int k = 0; k < m->top_k && k < 8; k++) {
        int eid = h_ids[k];
        if (eid >= 0 && eid < 128 && h_expert_gate[layer_idx][eid]) {
            g[n_valid] = h_expert_gate[layer_idx][eid];
            u[n_valid] = h_expert_up[layer_idx][eid];
            d[n_valid] = h_expert_down[layer_idx][eid];
            w[n_valid] = h_vals[k];
            n_valid++;
        }
    }
    g_experts_requested += (m->top_k < 8 ? m->top_k : 8);
    g_experts_executed  += n_valid;
    if (n_valid == 0) return;

    // 6. Routed experts (only valid ones)
    dim3 gus_grid(m->e_inter, n_valid);
    if (m->e_gqt == 0)
        mega8_q2k_gate_up_silu_kernel<<<gus_grid, T>>>(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],
            u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7], norm_buf, gate_buf, m->e_inter, m->e_in);
    else {
        for(int i=0;i<n_valid;i++){
            q3k_matvec_kernel<<<m->e_inter,T>>>(g[i],norm_buf,gate_buf+i*m->e_inter,m->e_inter,m->e_in);
            q3k_matvec_kernel<<<m->e_inter,T>>>(u[i],norm_buf,mega_temp,m->e_inter,m->e_in);
            swiglu_fuse_kernel<<<(m->e_inter+T-1)/T,T>>>(gate_buf+i*m->e_inter,mega_temp,m->e_inter);
        }
    }
    dim3 dn_grid(m->e_dr, n_valid);
    if (m->e_dqt == 0)
        mega8_q2k_down_kernel<<<dn_grid, T>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
            gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
    else
        mega8_q3k_down_kernel<<<dn_grid, T>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
            gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
    reduce8_and_accumulate_kernel<<<(m->e_dr+T-1)/T, T>>>(mega_temp, x,
        m->e_dr, n_valid, w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
}

// Launch JUST the preamble (rms_norm + router + topk + shared expert) for graph capture
static void moe_preamble_kernels(float* x, MoeLayerInfo* m, float* norm_buf, float* gate_buf, float* mega_temp) {
    int T = 256;
    rms_norm_kernel<<<1, T>>>(x, m->ffn_norm_w, norm_buf, m->hidden_dim);
    fp16_matvec_kernel<<<m->n_experts_total, T>>>(m->router_W, norm_buf, m->router_buf, m->n_experts_total, m->hidden_dim);
    topk_kernel<<<1, 1>>>(m->router_buf, m->router_bias, m->topk_vals_buf, m->topk_ids_buf, m->n_experts_total, m->top_k);
    if (m->sh_gate && !g_no_shared) {
        if (m->sh_gqt == 0)
            q2k_gate_up_silu_kernel<<<m->sh_inter, T>>>(m->sh_gate, m->sh_up, norm_buf, gate_buf, m->sh_inter, m->sh_in);
        else {
            q3k_matvec_kernel<<<m->sh_inter, T>>>(m->sh_gate, norm_buf, gate_buf, m->sh_inter, m->sh_in);
            q3k_matvec_kernel<<<m->sh_inter, T>>>(m->sh_up, norm_buf, mega_temp, m->sh_inter, m->sh_in);
            swiglu_fuse_kernel<<<(m->sh_inter+T-1)/T, T>>>(gate_buf, mega_temp, m->sh_inter);
        }
        if (m->sh_dqt == 0)
            q2k_down_accumulate_kernel<<<m->sh_dr, T>>>(m->sh_down, gate_buf, x, m->sh_dr, m->sh_di);
        else
            q3k_down_accumulate_kernel<<<m->sh_dr, T>>>(m->sh_down, gate_buf, x, m->sh_dr, m->sh_di);
    }
}

__declspec(dllexport) void reset_expert_counters() {
    g_experts_requested = 0; g_experts_executed = 0;
}
__declspec(dllexport) long long get_experts_requested() { return g_experts_requested; }
__declspec(dllexport) long long get_experts_executed()  { return g_experts_executed; }

__declspec(dllexport) int capture_moe_preamble(
    int layer_idx, float* x, float* norm_buf, float* gate_buf, float* mega_temp
) {
    if (layer_idx >= 78 || !moe_info[layer_idx].valid) return -1;
    MoeLayerInfo* m = &moe_info[layer_idx];
    int slot = 78 + layer_idx;
    cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed);
    moe_preamble_kernels(x, m, norm_buf, gate_buf, mega_temp);
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(0, &graph);
    if (err != cudaSuccess) return -1;
    if (g_graph_execs[slot]) {
        cudaGraphExecDestroy(g_graph_execs[slot]);
        g_graph_execs[slot] = 0;
    }
    err = cudaGraphInstantiate(&g_graph_execs[slot], graph, 0);
    cudaGraphDestroy(graph);
    return (err == cudaSuccess) ? 0 : -2;
}

// Graphed MoE: replay preamble graph → sync → mega8
__declspec(dllexport) void full_moe_layer_v2(
    float* x, int layer_idx,
    float* norm_buf, float* gate_buf, float* mega_temp
) {
    if (layer_idx >= 78 || !moe_info[layer_idx].valid) return;
    MoeLayerInfo* m = &moe_info[layer_idx];
    int T = 256;
    int slot = 78 + layer_idx;

    // 1. Preamble: graph replay OR individual launches. A captured graph is baked,
    //    so bisect mode must take the un-graphed path for the flag to take effect.
    if (g_graph_execs[slot] && !g_no_shared) {
        cudaGraphLaunch(g_graph_execs[slot], 0);
    } else {
        moe_preamble_kernels(x, m, norm_buf, gate_buf, mega_temp);
    }

    // 2. Sync + read router results
    cudaDeviceSynchronize();
    int h_ids[8]; float h_vals[8];
    cudaMemcpy(h_ids, m->topk_ids_buf, m->top_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals, m->topk_vals_buf, m->top_k * sizeof(float), cudaMemcpyDeviceToHost);

    // 3. Look up expert pointers (with NULL guard)
    const uint8_t* g[8]={0}, *u[8]={0}, *d[8]={0}; float w[8]={0};
    int n_valid = 0;
    for (int k = 0; k < m->top_k && k < 8; k++) {
        int eid = h_ids[k];
        if (eid < 0 || eid >= 128) continue;
        log_route(layer_idx, eid);
        if (h_expert_gate[layer_idx][eid]) {
            g[n_valid] = h_expert_gate[layer_idx][eid];
            u[n_valid] = h_expert_up[layer_idx][eid];
            d[n_valid] = h_expert_down[layer_idx][eid];
            w[n_valid] = h_vals[k];
            n_valid++;
        } else {
            log_miss(layer_idx, eid);
        }
    }
    g_experts_requested += (m->top_k < 8 ? m->top_k : 8);
    g_experts_executed  += n_valid;
    if (n_valid == 0) return;

    // 4. Routed experts. The v2 kernels do 32-bit loads into the Q2_K payload, so
    //    they need 4-byte-aligned bases; fall back to v1 if any expert is not.
    //    (Q3_K blocks are 110B = only 2-byte aligned, so its v2 reads qs bytewise
    //     and has no such requirement.)
    int T2 = 128;
    bool gu_aligned = true, dn_aligned = true;
    for (int i = 0; i < n_valid; i++) {
        if (((uintptr_t)g[i] & 3) || ((uintptr_t)u[i] & 3)) gu_aligned = false;
        if ((uintptr_t)d[i] & 3) dn_aligned = false;
    }
    if ((m->e_in & 15) || (m->e_di & 15)) { gu_aligned = false; dn_aligned = false; }

    dim3 gus_grid(m->e_inter, n_valid);
    if (m->e_gqt == 0) {
        if (gu_aligned)
            mega8_q2k_gus_v2_kernel<<<gus_grid, T2>>>(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],
                u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7], norm_buf, gate_buf, m->e_inter, m->e_in);
        else
            mega8_q2k_gate_up_silu_kernel<<<gus_grid, T>>>(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],
                u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7], norm_buf, gate_buf, m->e_inter, m->e_in);
    } else {
        for(int i=0;i<n_valid;i++){
            q3k_matvec_kernel<<<m->e_inter,T>>>(g[i],norm_buf,gate_buf+i*m->e_inter,m->e_inter,m->e_in);
            q3k_matvec_kernel<<<m->e_inter,T>>>(u[i],norm_buf,mega_temp,m->e_inter,m->e_in);
            swiglu_fuse_kernel<<<(m->e_inter+T-1)/T,T>>>(gate_buf+i*m->e_inter,mega_temp,m->e_inter);
        }
    }
    dim3 dn_grid(m->e_dr, n_valid);
    if (m->e_dqt == 0) {
        if (dn_aligned)
            mega8_q2k_down_v2_kernel<<<dn_grid, T2>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
        else
            mega8_q2k_down_kernel<<<dn_grid, T>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
    } else {
        if (dn_aligned)
            mega8_q3k_down_v2_kernel<<<dn_grid, T2>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
        else
            mega8_q3k_down_kernel<<<dn_grid, T>>>(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
                gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
    }
    reduce8_and_accumulate_kernel<<<(m->e_dr+T-1)/T, T>>>(mega_temp, x,
        m->e_dr, n_valid, w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
}

// ============================================================
// ZERO-SYNC MoE: device-side expert pointer table
// ============================================================

// GPU-resident pointer table: d_expert_tables[layer] -> [128*3] void* on device
static void** d_expert_tables[78] = {};

__declspec(dllexport) void upload_expert_table(int layer_idx, const void** host_ptrs) {
    if (layer_idx >= 78) return;
    if (!d_expert_tables[layer_idx])
        cudaMalloc(&d_expert_tables[layer_idx], 128 * 3 * sizeof(void*));
    cudaMemcpy(d_expert_tables[layer_idx], host_ptrs, 128 * 3 * sizeof(void*), cudaMemcpyHostToDevice);
}

// Lookup gate+up+silu: read expert pointers from device table via topk_ids
__global__ void mega_q2k_gus_lookup_kernel(
    const void** __restrict__ table, const int* __restrict__ topk_ids,
    const float* __restrict__ x, float* __restrict__ out,
    int n_rows, int in_features
) {
    int expert_local = blockIdx.y, row = blockIdx.x;
    if (row >= n_rows) return;
    int eid = topk_ids[expert_local];
    const uint8_t* gW = (const uint8_t*)table[eid * 3 + 0];
    const uint8_t* uW = (const uint8_t*)table[eid * 3 + 1];
    int bpr = (in_features+255)/256, byte_row = bpr*84;
    const uint8_t* gr = gW + (long long)row*byte_row;
    const uint8_t* ur = uW + (long long)row*byte_row;
    float ga=0.0f, ua=0.0f;
    int tq = in_features/4;
    for (int qi = threadIdx.x; qi < tq; qi += blockDim.x) {
        int e=qi*4, blk=e/256, pos=e%256, sub=pos/16, qo=16+pos/4;
        const uint8_t* gb=gr+blk*84;
        float gd=__half2float(*(const __half*)(gb+80)), gm=__half2float(*(const __half*)(gb+82));
        float gdl=gd*(float)(gb[sub]&0x0F), gml=gm*(float)(gb[sub]>>4);
        uint8_t gq=gb[qo];
        const uint8_t* ub=ur+blk*84;
        float ud=__half2float(*(const __half*)(ub+80)), um=__half2float(*(const __half*)(ub+82));
        float udl=ud*(float)(ub[sub]&0x0F), uml=um*(float)(ub[sub]>>4);
        uint8_t uq=ub[qo];
        float x0=x[e],x1=x[e+1],x2=x[e+2],x3=x[e+3];
        ga+=(gdl*(float)(gq&3)-gml)*x0+(gdl*(float)((gq>>2)&3)-gml)*x1
           +(gdl*(float)((gq>>4)&3)-gml)*x2+(gdl*(float)((gq>>6)&3)-gml)*x3;
        ua+=(udl*(float)(uq&3)-uml)*x0+(udl*(float)((uq>>2)&3)-uml)*x1
           +(udl*(float)((uq>>4)&3)-uml)*x2+(udl*(float)((uq>>6)&3)-uml)*x3;
    }
    for(int off=16;off>0;off/=2){ga+=__shfl_down_sync(0xffffffff,ga,off);ua+=__shfl_down_sync(0xffffffff,ua,off);}
    __shared__ float ws[64];
    int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    if(lane==0){ws[wid]=ga;ws[32+wid]=ua;}
    __syncthreads();
    if(wid==0){
        int nw=(blockDim.x+31)>>5;
        ga=(lane<nw)?ws[lane]:0.0f; ua=(lane<nw)?ws[32+lane]:0.0f;
        for(int off=16;off>0;off/=2){ga+=__shfl_down_sync(0xffffffff,ga,off);ua+=__shfl_down_sync(0xffffffff,ua,off);}
        if(lane==0){float sig=1.0f/(1.0f+expf(-ga)); out[expert_local*n_rows+row]=ga*sig*ua;}
    }
}

// Lookup down Q2K: read expert pointer from device table
__global__ void mega_q2k_down_lookup_kernel(
    const void** __restrict__ table, const int* __restrict__ topk_ids,
    const float* __restrict__ gate_buf, float* __restrict__ temp_out,
    int n_rows, int in_features, int inter_dim
) {
    int expert_local = blockIdx.y, row = blockIdx.x;
    if (row >= n_rows) return;
    int eid = topk_ids[expert_local];
    const uint8_t* dW = (const uint8_t*)table[eid * 3 + 2];
    const float* h = gate_buf + expert_local * inter_dim;
    int bpr=(in_features+255)/256;
    const uint8_t* rd=dW+(long long)row*bpr*84;
    float acc=0.0f;
    int tq=in_features/4;
    for(int qi=threadIdx.x;qi<tq;qi+=blockDim.x){
        int e=qi*4,blk=e/256,pos=e%256,sub=pos/16;
        const uint8_t* b=rd+blk*84;
        float d=__half2float(*(const __half*)(b+80)),dm=__half2float(*(const __half*)(b+82));
        float dl=d*(float)(b[sub]&0x0F),ml=dm*(float)(b[sub]>>4);
        uint8_t qb=b[16+pos/4];
        acc+=(dl*(float)(qb&3)-ml)*h[e]+(dl*(float)((qb>>2)&3)-ml)*h[e+1]
            +(dl*(float)((qb>>4)&3)-ml)*h[e+2]+(dl*(float)((qb>>6)&3)-ml)*h[e+3];
    }
    __shared__ float ws[32];
    acc=block_reduce(acc,ws);
    if(threadIdx.x==0) temp_out[expert_local * n_rows + row] = acc;
}

// Lookup down Q3K: read expert pointer from device table
__global__ void mega_q3k_down_lookup_kernel(
    const void** __restrict__ table, const int* __restrict__ topk_ids,
    const float* __restrict__ gate_buf, float* __restrict__ temp_out,
    int n_rows, int in_features, int inter_dim
) {
    int expert_local = blockIdx.y, row = blockIdx.x;
    if (row >= n_rows) return;
    int eid = topk_ids[expert_local];
    const uint8_t* dW = (const uint8_t*)table[eid * 3 + 2];
    const float* h = gate_buf + expert_local * inter_dim;
    int bpr=(in_features+255)/256;
    const uint8_t* rd=dW+(long long)row*bpr*110;
    float acc=0.0f;
    int tq=in_features/4;
    for(int qi=threadIdx.x;qi<tq;qi+=blockDim.x){
        int e=qi*4,blk=e/256,pos=e%256,sub=pos/16;
        const uint8_t* b=rd+blk*110;
        float d=__half2float(*(const __half*)(b+108));
        float sc[16]; q3k_unpack_scales(b+96,sc);
        float dl=d*sc[sub];
        uint8_t qb=b[32+pos/4];
        int q0=(qb&3),q1=((qb>>2)&3),q2=((qb>>4)&3),q3=((qb>>6)&3);
        int h0=(b[pos/8]>>(pos%8))&1,h1=(b[(pos+1)/8]>>((pos+1)%8))&1;
        int h2=(b[(pos+2)/8]>>((pos+2)%8))&1,h3=(b[(pos+3)/8]>>((pos+3)%8))&1;
        acc+=dl*(float)(q0-4+h0*4)*h[e]+dl*(float)(q1-4+h1*4)*h[e+1]
            +dl*(float)(q2-4+h2*4)*h[e+2]+dl*(float)(q3-4+h3*4)*h[e+3];
    }
    __shared__ float ws[32];
    acc=block_reduce(acc,ws);
    if(threadIdx.x==0) temp_out[expert_local * n_rows + row] = acc;
}

// Reduce with device-side weights (reads topk_vals from GPU, no host copy)
__global__ void reduce_accumulate_lookup_kernel(
    const float* __restrict__ temp_out, float* __restrict__ x,
    const float* __restrict__ topk_vals,
    int n_rows, int n_experts
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float s = 0.0f;
    for (int k = 0; k < n_experts; k++)
        s += topk_vals[k] * temp_out[k * n_rows + row];
    x[row] += s;
}

// ZERO-SYNC full MoE layer: no cudaDeviceSynchronize, no cudaMemcpy
__declspec(dllexport) void full_moe_layer_nosync(
    float* x, int layer_idx,
    float* norm_buf, float* gate_buf, float* mega_temp
) {
    if (layer_idx >= 78 || !moe_info[layer_idx].valid || !d_expert_tables[layer_idx]) return;
    MoeLayerInfo* m = &moe_info[layer_idx];
    void** table = d_expert_tables[layer_idx];
    int T = 256;

    // 1. RMS norm
    rms_norm_kernel<<<1, T>>>(x, m->ffn_norm_w, norm_buf, m->hidden_dim);

    // 2. Router (async — results stay on device)
    fp16_matvec_kernel<<<m->n_experts_total, T>>>(m->router_W, norm_buf, m->router_buf, m->n_experts_total, m->hidden_dim);
    topk_kernel<<<1, 1>>>(m->router_buf, m->router_bias, m->topk_vals_buf, m->topk_ids_buf, m->n_experts_total, m->top_k);

    // 3. Shared expert
    if (m->sh_gate) {
        if (m->sh_gqt == 0)
            q2k_gate_up_silu_kernel<<<m->sh_inter, T>>>(m->sh_gate, m->sh_up, norm_buf, gate_buf, m->sh_inter, m->sh_in);
        else {
            q3k_matvec_kernel<<<m->sh_inter, T>>>(m->sh_gate, norm_buf, gate_buf, m->sh_inter, m->sh_in);
            q3k_matvec_kernel<<<m->sh_inter, T>>>(m->sh_up, norm_buf, mega_temp, m->sh_inter, m->sh_in);
            swiglu_fuse_kernel<<<(m->sh_inter+T-1)/T, T>>>(gate_buf, mega_temp, m->sh_inter);
        }
        if (m->sh_dqt == 0)
            q2k_down_accumulate_kernel<<<m->sh_dr, T>>>(m->sh_down, gate_buf, x, m->sh_dr, m->sh_di);
        else
            q3k_down_accumulate_kernel<<<m->sh_dr, T>>>(m->sh_down, gate_buf, x, m->sh_dr, m->sh_di);
    }

    // 4. Routed experts — ZERO SYNC, device-side pointer lookup
    const void** ctable = (const void**)table;
    dim3 gus_grid(m->e_inter, m->top_k);
    if (m->e_gqt == 0)
        mega_q2k_gus_lookup_kernel<<<gus_grid, T>>>(ctable, m->topk_ids_buf, norm_buf, gate_buf, m->e_inter, m->e_in);
    else {
        // Q3K fallback: still need individual launches per expert (rare path)
        cudaDeviceSynchronize();
        int h_ids[8];
        cudaMemcpy(h_ids, m->topk_ids_buf, m->top_k*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i=0;i<m->top_k;i++){
            const uint8_t* gi = h_expert_gate[layer_idx][h_ids[i]];
            const uint8_t* ui = h_expert_up[layer_idx][h_ids[i]];
            q3k_matvec_kernel<<<m->e_inter,T>>>(gi,norm_buf,gate_buf+i*m->e_inter,m->e_inter,m->e_in);
            q3k_matvec_kernel<<<m->e_inter,T>>>(ui,norm_buf,mega_temp,m->e_inter,m->e_in);
            swiglu_fuse_kernel<<<(m->e_inter+T-1)/T,T>>>(gate_buf+i*m->e_inter,mega_temp,m->e_inter);
        }
    }
    dim3 dn_grid(m->e_dr, m->top_k);
    if (m->e_dqt == 0)
        mega_q2k_down_lookup_kernel<<<dn_grid, T>>>(ctable, m->topk_ids_buf, gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
    else
        mega_q3k_down_lookup_kernel<<<dn_grid, T>>>(ctable, m->topk_ids_buf, gate_buf, mega_temp, m->e_dr, m->e_di, m->e_inter);
    reduce_accumulate_lookup_kernel<<<(m->e_dr+T-1)/T, T>>>(mega_temp, x, m->topk_vals_buf, m->e_dr, m->top_k);
}

}  // extern "C"
