# UNTESTED: this machine has no GPU and Triton is not installed. Never run here. Shown for reading only (Day 154).
# Row softmax in Triton: one program per row; the whole row is loaded into registers/SRAM once (BLOCK >= n_cols), so HBM is
# read once and written once -- the "fused" kernel the CPU Day 153 3-pass version approximates.
import torch, triton, triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * out_stride + offs, e / tl.sum(e, axis=0), mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return y

if __name__ == "__main__":  # what you would check on a GPU machine
    x = torch.randn(4096, 1024, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))
