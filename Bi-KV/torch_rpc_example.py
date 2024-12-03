import torch.distributed.rpc as rpc
import multiprocessing as mp

from Worker.Worker import Worker
from Scheduler.LLMScheduler import LLMScheduler,model_params,PromptOrder
from inputGenerator.inputGenerator import LLMInput,InputPrompt
from DistributedStorage.kvcache import KVCache
from config import *

args.model_code = 'llm'
# args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games"
args.dataset_code = "games"

total_capacity = 20000000000  # 20GB
user_cache_ratio = 0.5
model_layers = model_params.get("num_layers")
vector_dim = model_params.get("head_size") * model_params.get("num_kv_heads")
my_cache = KVCache(total_capacity=total_capacity, user_cache_ratio=user_cache_ratio, model_layers=model_layers, vector_dim=vector_dim)

def read_cache_with_rref(prompt_rref,ind):
    prompt:InputPrompt = prompt_rref.to_here()
    prompt_order = PromptOrder(prompt)
    user_access_time = 0
    item_access_time = 0
    computation_cost = 0
    user_cache_miss_times = 0
    item_cache_miss_times = 0
    if prompt_order == "User History First":
        # print("User first")
        user_access_time += 1
        user_data = my_cache.get(cache_type='user', key=ind)
        computation_cost += sum([item.token_count for item in prompt.items])
        if user_data is None:
            user_cache_miss_times += 1
            my_cache.put(cache_type='user', key=ind, sequence_length=prompt.user_history_tokens)
            computation_cost += prompt.user_history_tokens
    else:
        # assert len(prompt.items) == self.item_num
        item_access_time += len(prompt.items)
        computation_cost += prompt.user_history_tokens
        # print("Item first")
        for i, item in enumerate(prompt.items):
            item_data = my_cache.get(cache_type='item', key=i+10000*ind)
            if item_data is None:
                item_cache_miss_times += 1
                my_cache.put(cache_type='item', key=i+10000*ind, sequence_length=item.token_count)
                computation_cost += item.token_count
    item = prompt.timestamp
    id = prompt.user_id
    worker_id = int(rpc.get_worker_info().name[-1]) + 1  # 获取当前 worker 的 id
    return {"id":id,"worker_id":worker_id,"user_cache_miss_times":user_cache_miss_times,\
            "item_cache_miss_times":item_cache_miss_times,"computation_cost":computation_cost,\
            "user_access_time":user_access_time,"item_access_time":item_access_time}

# 启动函数
def run_worker(rank, world_size):
    worker = Worker(rank,world_size)
    worker.start()
    worker.shutdown()

def run_scheduler(world_size):
    llm_input = LLMInput(20,500,args)
    llm_input.set_random("weighted")
    generate_res = llm_input.Generate(100)
    scheduler = LLMScheduler(worker_func=read_cache_with_rref,world_size=world_size)
    scheduler.start()
    scheduler.schedule_prompt_list(generate_res)
    scheduler.shutdown()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    processes = []
    world_size = 10
    # 启动 workers
    for rank in range(1, world_size):  # 假设有 2 个 worker
        p = mp.Process(target=run_worker, args=(rank, world_size))
        p.start()
        processes.append(p)

    # 启动 scheduler
    p = mp.Process(target=run_scheduler,args=(world_size,))
    p.start()
    processes.append(p)
    options = rpc.TensorPipeRpcBackendOptions(init_method='tcp://localhost:29500', num_worker_threads=256, rpc_timeout=100)
    for p in processes:
        p.join()






