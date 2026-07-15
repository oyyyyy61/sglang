from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.triton_backend import logit_capping_mod
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()


def _get_mnw(
    num_seqs: int,
    hratio: int,
    kv_len: int,
    page_block_size: int,
) -> tuple[int, int, int]:
    m_val = num_seqs * hratio
    if m_val > 32:
        tile_m, warps = 64, 4
    elif m_val > 16:
        tile_m, warps = 32, 2
    else:
        tile_m, warps = 16, 1

    if kv_len < 32:
        tile_n = 16
    elif kv_len < 64:
        tile_n = 32
    elif kv_len < 128:
        tile_n = 64
    else:
        tile_n = 128
    if tile_m == 64:
        tile_n = max(32, tile_n)

    # The vLLM ForkAttention kernel requires each tile to stay within one
    # paged-cache block for a warp group.
    while tile_n > 16 and tile_n // (warps * 32 // 8) > page_block_size:
        tile_n //= 2
    return tile_m, tile_n, warps


def _fork_attention_op_available() -> bool:
    try:
        import sgl_kernel  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to import sgl_kernel for ForkAttention: %s", exc)
        return False
    return hasattr(torch.ops, "sgl_kernel") and hasattr(
        torch.ops.sgl_kernel, "fork_attention"
    )


def _run_fork_attention(
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    split_out: torch.Tensor,
    split_lse: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_split_per_seq: torch.Tensor,
    query_tables: list[torch.Tensor],
    block_tables: list[torch.Tensor],
    num_seqs_per_ctas: list[torch.Tensor],
    cta_ranks: list[torch.Tensor],
    kv_in_ctas: list[torch.Tensor],
    mnw: list[int],
    max_split_per_seq: int,
    softmax_scale: float,
) -> None:
    torch.ops.sgl_kernel.fork_attention.default(
        out,
        softmax_lse,
        split_out,
        split_lse,
        q,
        k_cache,
        v_cache,
        num_split_per_seq,
        query_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max_split_per_seq,
        softmax_scale,
    )


class ForkAttnBackend(TritonAttnBackend):
    """Triton-compatible backend with an opt-in ForkAttention decode path.

    The first cut keeps SGLang's existing Triton metadata/build path and only
    swaps decode attention to ForkAttention when the runtime layout matches the
    vLLM kernel contract. Unsupported cases fall back to Triton.
    """

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner, *args, **kwargs):
        super().__init__(model_runner, *args, **kwargs)
        self._fork_attn_warned_reasons: set[str] = set()
        self._fork_attn_used_logged = False

    def _warn_once(self, reason: str) -> None:
        if reason in self._fork_attn_warned_reasons:
            return
        self._fork_attn_warned_reasons.add(reason)
        logger.warning("ForkAttention decode falls back to Triton: %s", reason)

    def _fallback_reason(
        self,
        q: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        logits_soft_cap: float,
        k_descale: float,
        v_descale: float,
        sinks: Optional[torch.Tensor],
    ) -> Optional[str]:
        if not _fork_attention_op_available():
            return "torch.ops.sgl_kernel.fork_attention is not registered"
        if not _is_cuda:
            return "CUDA is required"
        if self.use_mla:
            return "MLA attention is not supported"
        if self.dcp_size > 1:
            return "DCP attention is not supported"
        if forward_batch.batch_size <= 0:
            return "empty batch"
        if layer.attn_type != AttentionType.DECODER or layer.is_cross_attention:
            return "only decoder self-attention is supported"
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            return "sliding window is not supported"
        if sinks is not None:
            return "attention sinks are not supported"
        if logits_soft_cap != 0:
            return "logit soft cap is not supported"
        if k_descale != 1.0 or v_descale != 1.0:
            return "quantized KV descale is not supported"
        if q.dtype not in (torch.float16, torch.bfloat16):
            return f"query dtype {q.dtype} is not supported"
        if layer.qk_head_dim != layer.v_head_dim:
            return "qk head dim and v head dim differ"
        if layer.qk_head_dim not in (64, 128):
            return f"head dim {layer.qk_head_dim} is not supported"
        if layer.tp_q_head_num % layer.tp_k_head_num != 0:
            return "query heads must be divisible by KV heads"
        if self.page_size % 16 != 0:
            return f"page_size={self.page_size} is not divisible by 16"

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if k_cache.dim() != 4 or v_cache.dim() != 4:
            return "KV cache is not a 4D paged layout"
        if k_cache.shape != v_cache.shape:
            return "key/value cache shapes differ"
        if k_cache.shape[1] != self.page_size:
            return "KV cache page dimension does not match page_size"
        if k_cache.shape[2] != layer.tp_k_head_num:
            return "KV cache head count does not match layer"
        if k_cache.shape[3] != layer.qk_head_dim:
            return "KV cache head dim does not match layer"
        if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
            return "KV cache dtype must match query dtype"
        if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
            return "KV cache last dimension must be contiguous"
        return None

    def _build_fork_metadata(
        self,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        layer,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[int],
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens[:bs].to(torch.int32)
        max_seq_len = int(seq_lens.max().item())
        max_blocks = (max_seq_len + self.page_size - 1) // self.page_size
        hratio = layer.tp_q_head_num // layer.tp_k_head_num
        mnw = list(_get_mnw(1, hratio, max_seq_len, self.page_size))

        block_slots = self.req_to_token[
            forward_batch.req_pool_indices[:bs],
            : max_blocks * self.page_size : self.page_size,
        ]
        block_table = (block_slots // self.page_size).to(torch.int32).contiguous()

        query_table = torch.arange(bs, dtype=torch.int32, device=q.device).view(bs, 1)
        num_seqs_per_cta = torch.ones(bs, dtype=torch.int32, device=q.device)
        cta_rank = torch.zeros(bs, dtype=torch.int32, device=q.device)
        num_split_per_seq = torch.ones(bs, dtype=torch.int32, device=q.device)

        softmax_lse = torch.empty(
            (bs, layer.tp_q_head_num, 1),
            dtype=torch.float32,
            device=q.device,
        )
        split_out = torch.empty((0,), dtype=torch.float32, device=q.device)
        split_lse = torch.empty((0,), dtype=torch.float32, device=q.device)

        return (
            num_split_per_seq,
            [query_table],
            [block_table],
            [num_seqs_per_cta],
            [cta_rank],
            [seq_lens.contiguous()],
            mnw,
            1,
            softmax_lse,
            split_out,
            split_lse,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            if self.use_mla:
                if layer.k_scale is not None:
                    k.div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        reason = self._fallback_reason(
            q, layer, forward_batch, logits_soft_cap, k_descale, v_descale, sinks
        )
        if reason is None:
            if not self._fork_attn_used_logged:
                logger.info("ForkAttention decode path is enabled.")
                self._fork_attn_used_logged = True
            (
                num_split_per_seq,
                query_tables,
                block_tables,
                num_seqs_per_ctas,
                cta_ranks,
                kv_in_ctas,
                mnw,
                max_split_per_seq,
                softmax_lse,
                split_out,
                split_lse,
            ) = self._build_fork_metadata(forward_batch, q, layer)
            _run_fork_attention(
                o.view(-1, 1, layer.tp_q_head_num, layer.v_head_dim),
                softmax_lse,
                split_out,
                split_lse,
                q.view(-1, 1, layer.tp_q_head_num, layer.qk_head_dim),
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                num_split_per_seq,
                query_tables,
                block_tables,
                num_seqs_per_ctas,
                cta_ranks,
                kv_in_ctas,
                mnw,
                max_split_per_seq,
                layer.scaling,
            )
            return o

        self._warn_once(reason)
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            k_descale,
            v_descale,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
            has_mla=self.use_mla,
            use_pdl=self.use_pdl,
            page_size=self.page_size,
        )
        return o
