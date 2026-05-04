import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None  # type: ignore[assignment]
    tl    = None   # type: ignore[assignment]


# ---------------------------------------------------------------------------
# String constants — accessible without Triton installed
# ---------------------------------------------------------------------------

HANDWRITTEN_RMSNORM_SOURCE = """
# Hand-written Triton RMSNorm kernel (triton_kernels.py)
# Key differences from inductor auto-generated:
#   1. @triton.autotune — tunes BLOCK_SIZE per GPU (inductor uses static 512)
#   2. Explicit fp16->fp32 upcast with eviction hints (inductor uses default)
#   3. Backward kernel included (inductor auto-generates backward via autograd)
#   4. Single .to(tl.float16) downcast at store (same as inductor)
#   5. No separate cast kernel: dtype handled inside the JIT kernel

@triton.autotune([
    Config({'BLOCK_SIZE': 512},  num_warps=4),
    Config({'BLOCK_SIZE': 1024}, num_warps=8),
    Config({'BLOCK_SIZE': 2048}, num_warps=16),
], key=['N'])
@triton.jit
def _rms_norm_fwd_kernel(X, W, Y, stride_x, N, eps, BLOCK_SIZE: tl.constexpr):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x    = tl.load(X + row*stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w    = tl.load(W + cols, mask=mask, other=1.0)
    rrms = tl.rsqrt(tl.sum(x*x, axis=0)/N + eps)
    tl.store(Y + row*stride_x + cols, ((x*rrms)*w.to(tl.float32)).to(tl.float16), mask=mask)
"""

HANDWRITTEN_SOFTMAX_SOURCE = """
# Hand-written Triton Softmax kernel (triton_kernels.py)
# Key differences from inductor auto-generated:
#   1. @triton.autotune over BLOCK_SIZE and num_warps
#   2. Explicit 3-pass online algorithm (same as inductor)
#   3. For MoE router: N=num_experts fits in 1 block -> no inter-block sync

@triton.autotune([
    Config({'BLOCK_SIZE': 512},  num_warps=4),
    Config({'BLOCK_SIZE': 1024}, num_warps=8),
], key=['N'])
@triton.jit
def _softmax_fwd_kernel(X, Y, stride_x, N, BLOCK_SIZE: tl.constexpr):
    row   = tl.program_id(0)
    cols  = tl.arange(0, BLOCK_SIZE)
    mask  = cols < N
    x     = tl.load(X + row*stride_x + cols, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    tl.store(Y + row*stride_x + cols, x_exp / tl.sum(x_exp, axis=0), mask=mask)
"""


# ---------------------------------------------------------------------------
# Kernel implementations — only compiled when Triton is available
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
        ],
        key=["N"],
    )
    @triton.jit
    def _rms_norm_fwd_kernel(
        X, W, Y, stride_x, N, eps, BLOCK_SIZE: tl.constexpr,
    ):
        row  = tl.program_id(0)
        X   += row * stride_x
        Y   += row * stride_x
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x    = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w    = tl.load(W + cols, mask=mask, other=1.0)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
        y    = (x * rrms) * w.to(tl.float32)
        tl.store(Y + cols, y.to(tl.float16), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
        ],
        key=["N"],
    )
    @triton.jit
    def _rms_norm_bwd_kernel(
        DY, X, W, DX, DW, stride_x, N, eps, BLOCK_SIZE: tl.constexpr,
    ):
        row  = tl.program_id(0)
        X   += row * stride_x
        DY  += row * stride_x
        DX  += row * stride_x
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x    = tl.load(X  + cols, mask=mask, other=0.0).to(tl.float32)
        dy   = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w    = tl.load(W  + cols, mask=mask, other=1.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
        normed = x * rrms
        tl.atomic_add(DW + cols, dy * normed, mask=mask)
        dy_w = dy * w
        dx   = rrms * (dy_w - normed * tl.sum(dy_w * normed, axis=0) / N)
        tl.store(DX + cols, dx.to(tl.float16), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
        ],
        key=["N"],
    )
    @triton.jit
    def _softmax_fwd_kernel(
        X, Y, stride_x, N, BLOCK_SIZE: tl.constexpr,
    ):
        row   = tl.program_id(0)
        X    += row * stride_x
        Y    += row * stride_x
        cols  = tl.arange(0, BLOCK_SIZE)
        mask  = cols < N
        x     = tl.load(X + cols, mask=mask, other=-float("inf"))
        x_exp = tl.exp(x - tl.max(x, axis=0))
        tl.store(Y + cols, x_exp / tl.sum(x_exp, axis=0), mask=mask)

    class RMSNormTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
            orig = x.shape
            x2   = x.view(-1, x.shape[-1])
            M, N = x2.shape
            y    = torch.empty_like(x2)
            _rms_norm_fwd_kernel[(M,)](x2, weight, y, x2.stride(0), N, eps)
            ctx.save_for_backward(x2, weight)
            ctx.eps = eps
            return y.view(orig)

        @staticmethod
        def backward(ctx, dy):
            x2, w = ctx.saved_tensors
            M, N  = x2.shape
            dy2   = dy.view(M, N)
            dx    = torch.empty_like(x2)
            dw    = torch.zeros_like(w)
            _rms_norm_bwd_kernel[(M,)](dy2, x2, w, dx, dw, x2.stride(0), N, ctx.eps)
            return dx.view_as(dy), dw, None

    def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return RMSNormTriton.apply(x, weight, eps)

    def softmax_triton(x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2   = x.view(-1, x.shape[-1])
        M, N = x2.shape
        y    = torch.empty_like(x2)
        _softmax_fwd_kernel[(M,)](x2, y, x2.stride(0), N)
        return y.view(orig)

else:
    def rms_norm_triton(x, weight, eps=1e-6):      # type: ignore[misc]
        raise RuntimeError("Triton not installed — rms_norm_triton unavailable.")

    def softmax_triton(x):                          # type: ignore[misc]
        raise RuntimeError("Triton not installed — softmax_triton unavailable.")
