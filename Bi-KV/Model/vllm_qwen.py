# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors #, SamplerOutput
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from utils import is_pp_missing_parameter, make_layers
from vllm.utils import get_open_port
import flashinfer
import time

PREFIXLEN = 8192

SUFFIXLEN = 5120

class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        
        
        self.shared_prefix_len = PREFIXLEN
        self.nnz_qo = SUFFIXLEN
        self.kv_cache_block_size = 64
        self.max_kv_cache_blocks = 128
        
        self.shared_k_data_cpu = torch.randn(
            self.shared_prefix_len, num_kv_heads, self.head_dim, dtype=torch.float16, device="cpu"
        )
        
        self.shared_v_data_cpu = torch.randn(
                self.shared_prefix_len, num_kv_heads, self.head_dim, dtype=torch.float16, device="cpu"
        )        
        
        self.shared_k_data = torch.randn(
            self.shared_prefix_len, num_kv_heads, self.head_dim, dtype=torch.float16, device="cuda:0"
        )
        
        self.shared_v_data = torch.randn(
                self.shared_prefix_len, num_kv_heads, self.head_dim, dtype=torch.float16, device="cuda:0"
        )
    
        qo_indptr = torch.tensor(
            [0, self.nnz_qo], dtype=torch.int32, device="cuda:0"
        )
        
        paged_kv_indices = torch.arange(self.max_kv_cache_blocks).int().to("cuda:0")
        paged_kv_indptr = torch.tensor(
            [0, self.max_kv_cache_blocks], dtype=torch.int32, device="cuda:0"
        )
        # 1 <= paged_kv_last_page_len <= page_size
        paged_kv_last_page_len= torch.tensor(
            [self.kv_cache_block_size], dtype=torch.int32, device="cuda:0"
        )    
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        self.prefill_wrapper = flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        
        self.prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.kv_cache_block_size,
        )
    
        # self.attn = Attention(self.num_heads,
        #                       self.self.,
        #                       self.scaling,
        #                       num_kv_heads=self.num_kv_heads,
        #                       cache_config=cache_config,
        #                       quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # print(f"hidden state shape {hidden_states.shape}")
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # print(f"q shape {q.shape}")

        q, k = self.rotary_emb(positions, q, k)
        # q = torch.zeros(self.nnz_qo, self.num_heads, self.head_dim, dtype=torch.float16, device="cuda:0")
        q = q.view(self.nnz_qo, self.num_heads, self.head_dim).half()
        # attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output = self.prefill_wrapper.forward(
            q, self.shared_k_data, self.shared_v_data, kv_cache, causal=True
        )
        # print(f"attn_output shape {attn_output.shape}")

        attn_output = attn_output.view(self.nnz_qo, self.num_heads * self.head_dim)#.float()
        output, _ = self.o_proj(attn_output)
        # print(f"output shape {output.shape}")
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config),
            prefix=f"{prefix}.layers",
        )
        print(f"start layer {self.start_layer}, end layer {self.end_layer}")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # print(hidden_states.shape)
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = %s is less than "
                             "`num_hidden_layers` = %s. Please open an issue "
                             "to discuss this feature." % (
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        super().__init__()

        self.config = config

        self.quant_config = quant_config
        self.model = Qwen2Model(config, cache_config, quant_config)

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        # attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        # hidden_states = self.model(input_ids, positions, kv_caches,
                                #    attn_metadata, intermediate_tensors)
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   intermediate_tensors)        
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[SamplerOutput]:
    #     next_tokens = self.sampler(logits, sampling_metadata)
    #     return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


def init_worker_distributed_environment(
    # parallel_config: ParallelConfig,
    # rank: int,
    distributed_init_method: Optional[str] = None,
    # local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    # set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # init_distributed_environment(parallel_config.world_size, rank,
    #                              distributed_init_method, local_rank)

    # ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
    #                                   parallel_config.pipeline_parallel_size)
    set_custom_all_reduce(not True)

    # init_distributed_environment(1, 0,
    #                              distributed_init_method, 0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        local_rank=0)
    ensure_model_parallel_initialized(1, 1)


# num_layers = 32
# num_qo_heads = 64
# num_kv_heads = 16
# head_dim = 128
max_kv_cache_blocks = 128
# page_size = 16

hidden_size = 1536
intermediate_size = 8960
num_hidden_layers = 28
num_attention_heads = 12
num_key_value_heads = 2
head_dim = hidden_size // num_attention_heads
batch_size = 1
kv_cache_block_size = 64
max_kv_cache_blocks = 128
if __name__ == '__main__':
    init_worker_distributed_environment()
    torch.set_default_dtype(torch.float16)
    model_config = Qwen2Config(hidden_size = hidden_size,
                                intermediate_size = intermediate_size,
                                num_hidden_layers = num_hidden_layers,
                                num_attention_heads = num_attention_heads,
                                num_key_value_heads = num_key_value_heads)
    kv_caches = [
        torch.randn(
        max_kv_cache_blocks, 2, kv_cache_block_size, num_key_value_heads, head_dim, dtype=torch.float16, device="cuda:0"
        ) for _ in range(num_hidden_layers)
    ]

    cache_config = CacheConfig(kv_cache_block_size, 1.0, 1, "auto")
    
    QwenModel = Qwen2ForCausalLM(model_config, cache_config).to("cuda:0")
    print("start forward")
    
    nnz_qo = SUFFIXLEN
    shared_prefix_len = PREFIXLEN
    input_ids = torch.zeros(nnz_qo).int().to("cuda:0")
    
    # print(input_ids)
    # exit(0)
    positions = torch.arange(nnz_qo).long().to("cuda:0")+shared_prefix_len
    # print(positions)
    for i in range(10):
        output = QwenModel(input_ids, positions, kv_caches)    
    
    start_time = time.time()
    for i in range(10):
        output = QwenModel(input_ids, positions, kv_caches)
    print("latency: {}s".format((time.time()-start_time)/10))
    print(output.shape)
    