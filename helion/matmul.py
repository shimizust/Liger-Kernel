import torch, helion, helion.language as hl
import os

# Runs in eager mode if want to print out intermediate results
os.environ["HELION_INTERPRET"] = "1"

# @helion.kernel()
@helion.kernel(config=helion.Config(block_sizes=[32, 256, 64], indexing=['pointer', 'pointer', 'pointer'], l2_groupings=[4], load_eviction_policies=['', 'last'], loop_orders=[[1, 0]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, False], range_num_stages=[0, 3], range_unroll_factors=[0, 0], range_warp_specializes=[]), static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    i = 0
    for tile_m, tile_n in hl.tile([m, n]):
        print(tile_m, tile_n)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            print(i)
            i += 1
        out[tile_m, tile_n] = acc
    return out

if __name__ == "__main__":
    x = torch.randn(10024, 10024, device="cuda")
    y = torch.randn(10024, 10024, device="cuda")
    out = matmul(x, y)
    print(out)