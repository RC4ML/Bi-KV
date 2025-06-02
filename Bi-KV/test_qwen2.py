import torch
from config import *
from Model.qwen2 import process_task_info, prepare_attention_meta, Qwen2ForCausalLM, AttentionMetadata, token_shape, model_params
from transformers import Qwen2Config
from vllm.config import CacheConfig
import time
from protos import TaskInfo_pb2,TaskInfo_pb2_grpc

hidden_size = 1536
intermediate_size = 8960
num_hidden_layers = 28
num_attention_heads = 12
num_kv_heads = 2
head_dim = hidden_size // num_attention_heads
kv_cache_block_size = 16
max_kv_cache_blocks = 10240

task_info_list = [
                TaskInfo_pb2.TaskInfo(
                    request_id = 1,
                    id = -1,
                    infer_worker = 0,
                    token_num = 6144,
                    index = -1,
                    type = 'compute',
                )
    # {"request_id": 1, "id": -1, "recv_worker": 0, "token_num": 1024, "data_length": -1, "index": -1, "type": "compute"},
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
    

    prepare_time = time.time()
    cache_miss_dict = {}
    queried_task_info_list = process_task_info(task_info_list, cache_miss_dict)
    print("start forward")

    attn_metadata, cached_tokens = prepare_attention_meta(queried_task_info_list, kv_cache_block_size, max_kv_cache_blocks, device)
    print(f"prepare time: {(time.time()-prepare_time)}")

    input_ids = torch.zeros(attn_metadata.nnz_qo).int().to(device)
    positions = torch.arange(attn_metadata.nnz_qo).long().to(device)

    ## warmup
    for i in range(10):
        output = QwenModel(input_ids, positions, kv_caches, attn_metadata)    
    
    start_time = time.time()
    for i in range(10):
        output = QwenModel(input_ids, positions, kv_caches, attn_metadata)
    print("compute latency: {}s".format((time.time()-start_time)/10))
    print(f"communication latency: {(cached_tokens*num_kv_heads*head_dim*num_hidden_layers*2*2)/(20*1000*1000*1000)}s")
    
