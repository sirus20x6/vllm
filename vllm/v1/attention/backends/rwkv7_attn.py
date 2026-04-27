# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class Rwkv7AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RWKV7_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Rwkv7AttentionMetadataBuilder"]:
        return Rwkv7AttentionMetadataBuilder


@dataclass
class Rwkv7AttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_actual_tokens: int

    # Spec-decode counts. ``num_spec_decode_tokens`` is the total number of
    # tokens across the spec sequences (= sum of ``1 + num_spec`` across
    # spec sequences). Both are 0 when speculative decoding is disabled or
    # not active for this batch.
    num_spec_decodes: int = 0
    num_spec_decode_tokens: int = 0

    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None

    # Cache block-table for the recurrent state.
    # - In default mode (mamba_cache_mode != 'all', no spec): shape [batch],
    #   one cache block per active sequence; state is overwritten in place.
    # - In 'all' mode (prefix caching): shape [batch, max_blocks]; each
    #   sequence may use multiple blocks (one per ``mamba_block_size``
    #   tokens), and ``block_idx_*_*`` index into the second dim.
    state_indices_tensor: torch.Tensor | None = None

    # Boolean mask over prefill sequences only: True when the sequence has
    # cached state to load. None when num_prefills == 0.
    has_initial_state: torch.Tensor | None = None

    # Prefix-caching machinery (populated only when mamba_cache_mode == 'all').
    is_mamba_cache_all: bool = False
    mamba_block_size: int | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    num_computed_tokens_p: torch.Tensor | None = None

    # Resolved per-request cache slots for non-spec sequences (decodes +
    # prefills). 1D, length = num_decodes + num_prefills. Cudagraph-stable
    # under prefix caching.
    read_slot: torch.Tensor | None = None
    write_slot: torch.Tensor | None = None

    # Spec-decode slot tables. ``spec_state_indices_tensor`` has shape
    # ``[num_spec_decodes, num_spec + 1]`` and is consumed by the recurrent
    # kernel's ``ssm_state_indices`` argument: token ``t`` of spec sequence
    # ``i`` writes its post-state to cache block ``spec_state_indices_tensor
    # [i, t]``. Read-side initial-state slot for spec sequence ``i`` is
    # ``spec_state_indices_tensor[i, num_accepted_tokens[i] - 1]``.
    spec_state_indices_tensor: torch.Tensor | None = None
    non_spec_state_indices_tensor: torch.Tensor | None = None
    spec_query_start_loc: torch.Tensor | None = None
    non_spec_query_start_loc: torch.Tensor | None = None
    spec_sequence_masks: torch.Tensor | None = None
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None


class Rwkv7AttentionMetadataBuilder(AttentionMetadataBuilder[Rwkv7AttentionMetadata]):
    reorder_batch_threshold: int = 1

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        if vllm_config.cache_config.mamba_cache_mode == "all":
            # Prefix caching path is not yet cudagraph-safe for the spec
            # tables; fall back to eager when both flags are on.
            self._cudagraph_support = AttentionCGSupport.NEVER

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        max_seqs = vllm_config.scheduler_config.max_num_seqs
        self.decode_cudagraph_max_bs: int = max_seqs * (self.num_spec + 1)
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # Persistent buffers used as cudagraph-stable destinations. The
        # `read_slot` / `write_slot` pair covers the prefix-cache decode
        # path; the `spec_*` / `non_spec_*` set covers the spec-decode
        # path. When the corresponding feature is disabled they sit unused.
        self.read_slot: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.write_slot: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.bool, device=device
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,), dtype=torch.int32, device=device
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,), dtype=torch.int32, device=device
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> Rwkv7AttentionMetadata:
        m = common_attn_metadata
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        seq_lens = m.seq_lens

        is_mamba_cache_all = self.vllm_config.cache_config.mamba_cache_mode == "all"
        mamba_block_size = self.kv_cache_spec.block_size if is_mamba_cache_all else None

        block_idx_last_computed_token: torch.Tensor | None = None
        block_idx_first_scheduled_token: torch.Tensor | None = None
        block_idx_last_scheduled_token: torch.Tensor | None = None
        num_computed_tokens_p: torch.Tensor | None = None
        read_slot: torch.Tensor | None = None
        write_slot: torch.Tensor | None = None

        # Block-table tensor regardless of mode (we slice it below).
        if is_mamba_cache_all:
            block_table_tensor = m.block_table_tensor
            num_computed_tokens = m.compute_num_computed_tokens()
            block_idx_last_computed_token = torch.clamp(
                cdiv(num_computed_tokens, mamba_block_size) - 1, min=0
            )
            block_idx_first_scheduled_token = (
                cdiv(num_computed_tokens + 1, mamba_block_size) - 1
            )
            block_idx_last_scheduled_token = torch.clamp(
                cdiv(seq_lens, mamba_block_size) - 1, min=0
            )
            state_indices_tensor = block_table_tensor
            read_slot = block_table_tensor.gather(
                1, block_idx_last_computed_token.long().unsqueeze(1)
            ).squeeze(1)
            write_slot = block_table_tensor.gather(
                1, block_idx_last_scheduled_token.long().unsqueeze(1)
            ).squeeze(1)
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )
            state_indices_tensor = block_table_tensor[:, 0]
            read_slot = state_indices_tensor
            write_slot = state_indices_tensor

        # Detect spec sequences (vllm passes ``num_decode_draft_tokens_cpu``
        # with one entry per scheduled sequence; values >= 0 indicate
        # speculative decoding is active for that sequence).
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks: torch.Tensor | None = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        spec_state_indices_tensor: torch.Tensor | None = None
        non_spec_state_indices_tensor: torch.Tensor | None = None
        spec_query_start_loc: torch.Tensor | None = None
        non_spec_query_start_loc: torch.Tensor | None = None
        spec_token_indx: torch.Tensor | None = None
        non_spec_token_indx: torch.Tensor | None = None
        num_spec_decode_tokens = 0

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m,
                    decode_threshold=self.reorder_batch_threshold,
                    treat_short_extends_as_decodes=False,
                )
            )
            non_spec_state_indices_tensor = state_indices_tensor
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu: torch.Tensor | None = query_start_loc_cpu
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = int((non_spec_query_lens_cpu == 1).sum().item())
            num_zero_len = int((non_spec_query_lens_cpu == 0).sum().item())
            num_prefills = (
                non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            )
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                int(non_spec_query_lens_cpu.sum().item()) - num_decode_tokens
            )
            num_spec_decode_tokens = (
                int(query_lens_cpu.sum().item())
                - num_prefill_tokens
                - num_decode_tokens
            )

            # Match GDN: when spec decodes coexist with non-spec decodes,
            # treat the non-spec decodes as 1-token prefills so a single
            # kernel dispatch handles them.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                # Pure spec batch.
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    int(query_start_loc_cpu[-1].item()),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                # Mixed: split tokens by spec_sequence_masks.
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    int(query_lens.size(0)) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
            else:
                has_initial_state = has_initial_state[num_decodes:]
            if is_mamba_cache_all:
                if spec_sequence_masks is not None:
                    num_computed_tokens_p = context_lens_tensor[~spec_sequence_masks]
                else:
                    num_computed_tokens_p = context_lens_tensor[num_decodes:]

        # Pre-allocate cudagraph-stable buffers when the captured shape
        # matches. We capture three patterns:
        #  (a) prefix-cache uniform decode (existing): copies read/write_slot
        #  (b) spec-decode pure-spec: copies spec_*
        #  (c) plain uniform decode without spec or prefix cache: skip.
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            and spec_sequence_masks is not None
        ):
            batch_size = m.num_actual_tokens

            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]
            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)
        elif (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            batch_size = m.num_actual_tokens
            # Existing prefix-cache stable buffers: read_slot + write_slot.
            self.read_slot[:num_decodes].copy_(
                read_slot[:num_decodes], non_blocking=True
            )
            read_slot = self.read_slot[: m.num_actual_tokens]
            read_slot[num_decodes:].fill_(PAD_SLOT_ID)

            self.write_slot[:num_decodes].copy_(
                write_slot[:num_decodes], non_blocking=True
            )
            write_slot = self.write_slot[: m.num_actual_tokens]
            write_slot[num_decodes:].fill_(PAD_SLOT_ID)

            # Mirror non_spec_state_indices_tensor for the spec-decode path
            # (so the mixer can use a single read path).
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                : m.num_actual_tokens
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        return Rwkv7AttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=state_indices_tensor,
            has_initial_state=has_initial_state,
            is_mamba_cache_all=is_mamba_cache_all,
            mamba_block_size=mamba_block_size,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            num_computed_tokens_p=num_computed_tokens_p,
            read_slot=read_slot,
            write_slot=write_slot,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
        )
