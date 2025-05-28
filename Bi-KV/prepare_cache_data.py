from tqdm import tqdm
from config import *

args.model_code = 'llm'
args.dataset_code = 'books'
args.llm_base_model = "/share/nfs/models/Llama-2-7b-hf"
args.llm_base_tokenizer = "/share/nfs/models/Llama-2-7b-hf"
args.llm_retrieved_path = f"/share/nfs/sunjie/{args.dataset_code}"

from inputGenerator.inputGenerator import LLMInput
input_generator = LLMInput(100, 5, args,10)

import json
timestamp_map_path = f'/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/timestep_map.json'
with open(timestamp_map_path, 'r') as f:
    time_step_map = json.load(f)

input_prompts = []
save_item_prepare_data = {}
save_user_prepare_data = {}

for i in tqdm(range(len(input_generator.dataset)), total=len(input_generator.dataset), desc="Processing prompts"):
    prompt = input_generator.access_index(i)
    save_user_prepare_data[prompt.user_id + 2000000] = prompt.user_history_tokens
    for item in prompt.items:
        save_item_prepare_data[item.item_id] = item.token_count


str_item_data = {str(k): v for k, v in save_item_prepare_data.items()}

print(f"保存文件到 ./data/{args.dataset_code}/prepare_cache_data_item_all.json")

with open(f'./data/{args.dataset_code}/prepare_cache_data_item_all.json', 'w') as f:
    json.dump(str_item_data, f)


str_user_data = {str(k): v for k, v in save_user_prepare_data.items()}

print(f"保存文件到 ./data/{args.dataset_code}/prepare_cache_data_user_all.json")

with open(f'./data/{args.dataset_code}/prepare_cache_data_user_all.json', 'w') as f:
    json.dump(str_user_data, f)