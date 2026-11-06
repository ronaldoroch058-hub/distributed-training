import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__=["ScaledDotProductAttentionWrapper"]

class ScaledDotProductAttentionWrapper(torch.nn.module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self,
            q:torch.Tensor,
            k:torch.Tensor, 
            v:torch.Tensor,
            *,
            scale: float | None = None,
    ) -> torch.Tensor:
        v_head_dim = v.shape[-1]
        q_head_dim = q.shape[-1]


        tp_spec = None
        if isinstance(q, DTensor):
            assert isinstance(k, DTensor) and isinstance(v, DTensor)
            tp_spec = (q.device_mesh, q.placements)
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

            if v_head_dim < q_head_dim:
                v = F.pad(v, (0, q_head_dim - v_head_dim))

            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION
                    SDPBackend.EFFICIENT_ATTENTION
                    SDPBackend.CUDNN_ATTENTION
                ],
                set_priority = True,
            ):


                 out = F.scaled_dot_product_attention(q, k, v, scale = scale, is_casual = True)
            if out.shape [-1] != v_head_dim:
                out = out[..., :v_head_dim]

            if tp_spec is not None:
                out = DTensor.from_local(out, tp_spec[0], tp_spec[1], run_check = False)


            return out 