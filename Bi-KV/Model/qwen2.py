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

# from vllm.attention import Attention
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from torch.nn import Embedding
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors #, SamplerOutput
from Model.utils import make_layers

import flashinfer
import time
from collections import defaultdict

class AttentionMetadata():
    def __init__(self, nnz_qo, qo_indptr, kv_indptr, kv_indices, kv_last_page_len):
        self.nnz_qo = nnz_qo
        self.qo_indptr = qo_indptr
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_len = kv_last_page_len

class MergedColumnLinear(nn.Module):
    def __init__(self, hidden_size, output_sizes, bias=False, quant_config=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes

        # Create independent linear layers for the merged outputs
        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, out_size, bias=bias)
            for out_size in output_sizes
        ])

        # Optionally handle quantization config
        self.quant_config = quant_config

    def forward(self, x):
        # Apply each linear layer and concatenate the outputs
        outputs = [layer(x) for layer in self.linear_layers]
        return torch.cat(outputs, dim=-1)

class RowLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=False, quant_config=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.quant_config = quant_config

    def forward(self, x):
        return self.linear(x)

class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x) ## check this
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 prefill_wrapper,
                 device,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_kv_heads = max(1, num_kv_heads)
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        # Standard linear layers for Q, K, and V
        self.q_proj = nn.Linear(hidden_size, self.total_num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.total_num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.device = device
        self.prefill_wrapper=prefill_wrapper
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q, k = self.rotary_emb(positions, q, k)

        assert (q.shape[0] == attn_metadata.nnz_qo)
        self.prefill_wrapper.begin_forward(
            attn_metadata.qo_indptr,
            attn_metadata.kv_indptr,
            attn_metadata.kv_indices,
            attn_metadata.kv_last_page_len,
            self.total_num_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
        )
        attn_output = self.prefill_wrapper.forward(
            q.contiguous().view(-1, self.total_num_heads, self.head_dim), kv_cache
        )
        attn_output = attn_output.view(-1, self.total_num_heads * self.head_dim)
        output = self.o_proj(attn_output)
        
        # print(f"output shape {output.shape}")
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        prefill_wrapper,
        device,
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
            prefill_wrapper=prefill_wrapper,
            device=device,
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
        device,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayer(prefill_wrapper = self.prefill_wrapper,
                                            device=device,
                                             config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config),
            prefix=f"{prefix}.layers",
        )
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
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None

        # Loop through all layers without PP logic
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        # Apply the final normalization
        hidden_states = self.norm(hidden_states, residual)

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
        device,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.model = Qwen2Model(device=device, config=config)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = nn.Linear(config.vocab_size,
                                          config.hidden_size)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata,
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

def process_task_info(task_info_list):
    """
    整合 task_info_list，统计 cache hit 和 cache miss+compute 的 token 总数。

    Args:
    - task_info_list (list[dict]): 包含 task_info 的列表。
    - cache_hit_fn (function): 一个假定的函数，用于判断 task_info 是否是 cache hit。

    Returns:
    - list[dict]: 重新构造成的列表，包含每个 request_id 的统计数据。
    """
    # 用于存储每个 request_id 的统计数据
    request_stats = defaultdict(lambda: {"cached_tokens": 0, "recomputing_tokens": 0})

    for task_info in task_info_list:
        request_id = task_info["request_id"]
        token_num = task_info["token_num"]
        task_type = task_info["type"]

        if task_type == "user cache" or task_type == "item cache":
            if True:  # 假定的函数，用于判断 cache 是否命中
                request_stats[request_id]["cached_tokens"] += token_num
            else:  # cache miss 处理
                request_stats[request_id]["recomputing_tokens"] += token_num
        elif task_type == "compute":
            # compute 类型的 token_num 全部算作 "recomputing_tokens"
            request_stats[request_id]["recomputing_tokens"] += token_num

    # 将统计结果转化为列表
    queried_task_info_list = [
        {"request_id": req_id, **stats} for req_id, stats in request_stats.items()
    ]

    return queried_task_info_list

def prepare_attention_meta(
    queried_task_info_list,
    kv_cache_block_size,
    max_kv_cache_blocks,
    device
):    
    batch_size = len(queried_task_info_list)
    kv_seq_lens = []
    q_seq_lens = []
    total_communication_cost = 0
    for taskinfo in queried_task_info_list:
        kv_seq_lens.append(taskinfo['cached_tokens'] + taskinfo['recomputing_tokens'])
        # q_seq_lens.append(taskinfo['cached_tokens'] + taskinfo['recomputing_tokens'])
        q_seq_lens.append(taskinfo['recomputing_tokens'])

        total_communication_cost += taskinfo['cached_tokens']
    # Compute qo_indptr 
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(torch.tensor(q_seq_lens, dtype=torch.int32, device=device), dim=0)
    nnz_qo = qo_indptr[-1]
    # Calculate paged_kv_indptr and paged_kv_last_page_len
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    for i, size in enumerate(kv_seq_lens):
        full_pages = size // kv_cache_block_size  # Number of full pages
        last_page = size % kv_cache_block_size   # Tokens in the last (partial) page
        paged_kv_indptr[i + 1] = paged_kv_indptr[i] + full_pages + (1 if last_page > 0 else 0)
        paged_kv_last_page_len[i] = last_page if last_page > 0 else kv_cache_block_size       
    assert paged_kv_indptr[-1] <= max_kv_cache_blocks

    paged_kv_indices = torch.arange(paged_kv_indptr[-1], dtype=torch.int32, device=device)

    return AttentionMetadata(nnz_qo, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len), total_communication_cost

hidden_size = 1536
intermediate_size = 8960
num_hidden_layers = 28
num_attention_heads = 12
num_kv_heads = 2
head_dim = hidden_size // num_attention_heads
kv_cache_block_size = 16
max_kv_cache_blocks = 10240

task_info_list = [
    {"request_id": 1, "id": 1, "recv_worker": 0, "token_num": 682, "data_length": 50, "index": 0, "type": "user cache"},
    {"request_id": 1, "id": -1, "recv_worker": 0, "token_num": 25840, "data_length": -1, "index": -1, "type": "compute"},
    # {"request_id": 2, "id": 2, "recv_worker":  0, "token_num": 150, "data_length": 75, "index": 0, "type": "item cache"},
    # {"request_id": 2, "id": 3, "recv_worker":  0, "token_num": 105, "data_length": 105, "index": 1, "type": "item cache"},
    # {"request_id": 2, "id": 4, "recv_worker":  0, "token_num": 115, "data_length": 100, "index": 2, "type": "item cache"},
    # {"request_id": 2, "id": -1, "recv_worker": 0, "token_num": 250, "data_length": -1, "index": -1, "type": "compute"},
    # {"request_id": 3, "id": 2, "recv_worker":  0, "token_num": 150, "data_length": 75, "index": 0, "type": "item cache"},
    # {"request_id": 3, "id": 3, "recv_worker":  0, "token_num": 105, "data_length": 105, "index": 1, "type": "item cache"},
    # {"request_id": 3, "id": 4, "recv_worker":  0, "token_num": 115, "data_length": 100, "index": 2, "type": "item cache"},
    # {"request_id": 3, "id": -1, "recv_worker": 0, "token_num": 250, "data_length": -1, "index": -1, "type": "compute"},
    # {"request_id": 4, "id": 2, "recv_worker":  0, "token_num": 150, "data_length": 75, "index": 0, "type": "item cache"},
    # {"request_id": 4, "id": 3, "recv_worker":  0, "token_num": 105, "data_length": 105, "index": 1, "type": "item cache"},
    # {"request_id": 4, "id": 4, "recv_worker":  0, "token_num": 115, "data_length": 100, "index": 2, "type": "item cache"},
    # {"request_id": 4, "id": 2, "recv_worker":  0, "token_num": 150, "data_length": 75, "index": 0, "type": "item cache"},
    # {"request_id": 4, "id": 3, "recv_worker":  0, "token_num": 105, "data_length": 105, "index": 1, "type": "item cache"},
    # {"request_id": 4, "id": 4, "recv_worker":  0, "token_num": 115, "data_length": 100, "index": 2, "type": "item cache"},
    # {"request_id": 4, "id": -1, "recv_worker": 0, "token_num": 250, "data_length": -1, "index": -1, "type": "compute"},
    # {"request_id": 5, "id": 2, "recv_worker":  0, "token_num": 150, "data_length": 75, "index": 0, "type": "item cache"},
    # {"request_id": 5, "id": 3, "recv_worker":  0, "token_num": 105, "data_length": 105, "index": 1, "type": "item cache"},
    # {"request_id": 5, "id": 4, "recv_worker":  0, "token_num": 115, "data_length": 100, "index": 2, "type": "item cache"},
    # {"request_id": 5, "id": -1, "recv_worker": 0, "token_num": 250, "data_length": -1, "index": -1, "type": "compute"},
]

# task_info_list = [
#     {"request_id": 1, "id": 1, "recv_worker": 0, "token_num": 8192, "data_length": 50, "index": 0, "type": "user cache"},
#     {"request_id": 1, "id": -1, "recv_worker": 0, "token_num": 1024, "data_length": -1, "index": -1, "type": "compute"},
# ]

model_params = {
    "head_size": 128,
    "num_q_heads": 12, 
    "num_kv_heads": 2,      
    "num_layers": 28 
}

token_shape = (model_params['head_size'],
               model_params['num_kv_heads'],
               model_params['num_layers'],
               2)

if __name__ == '__main__':
    device = f"cuda:0"
    torch.set_default_dtype(torch.float16)
    model_config = Qwen2Config(hidden_size = hidden_size,
                                intermediate_size = intermediate_size,
                                num_hidden_layers = num_hidden_layers,
                                num_attention_heads = num_attention_heads,
                                num_key_value_heads = num_kv_heads)
    kv_caches = [
        torch.randn(
        max_kv_cache_blocks, 2, kv_cache_block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
        ) for _ in range(num_hidden_layers)
    ]
    cache_config = CacheConfig(kv_cache_block_size, 1.0, 1, "auto")
    QwenModel = Qwen2ForCausalLM(device, model_config, cache_config).to(device)
    
    print("start forward")

    prepare_time = time.time()
    queried_task_info_list = process_task_info(task_info_list)
    attn_metadata, cached_tokens = prepare_attention_meta(queried_task_info_list, kv_cache_block_size, max_kv_cache_blocks, device)

    input_ids = torch.zeros(attn_metadata.nnz_qo).int().to(device)
    positions = torch.arange(attn_metadata.nnz_qo).long().to(device)
    print(f"prepare time: {(time.time()-prepare_time)}")

    ## warmup
    for i in range(10):
        output = QwenModel(input_ids, positions, kv_caches, attn_metadata)    
    
    start_time = time.time()
    for i in range(10):
        output = QwenModel(input_ids, positions, kv_caches, attn_metadata)
    print("compute latency: {}s".format((time.time()-start_time)/10))
    print(f"communication latency: {(cached_tokens*num_kv_heads*head_dim*num_hidden_layers*2*2)/(20*1000*1000*1000)}s")
    
