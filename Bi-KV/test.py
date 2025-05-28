from typing import List
from config import *
from datasets import dataset_factory
from dataloader import LLMDataloader
from inputGenerator.inputGenerator import LLMInput,InputPrompt
import json
import pickle
# from huggingface_hub import login
# login()
args.model_code = 'llm'
args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
# args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games"
args.dataset_code = "books"
args.llm_base_model = "/share/nfs/models/Llama-2-7b-hf"
args.llm_base_tokenizer = "/share/nfs/models/Llama-2-7b-hf"
set_template(args)

def show_generate_res(res_list:List[InputPrompt]):
    for i in res_list:
        print(f"User ID: {i.user_id}\n\
User History Tokens: {i.user_history_tokens}\n\
Items: {i.items}\n\
Timestamp: {i.timestamp}\n\
Item Count: {len(i.items)}")
        


with open(f'./data/{args.dataset_code}/prepare_cache_data_user_all.json', 'r') as f:
    load_data_user = json.load(f)

with open(f'./data/{args.dataset_code}/prepare_cache_data_item_all.json', 'r') as f:
    load_item_user = json.load(f)

timestamp_map_path = f'/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/timestep_map.json'
with open(timestamp_map_path, 'r') as f:
    time_step_map = json.load(f)

item_access_path = f"/share/nfs/wsh/Bi-KV/Bi-KV/data/{args.dataset_code}/user_candidate.pickle"
with open(item_access_path, 'rb') as pickle_file:
    user_candidate_dict = pickle.load(pickle_file)

llm_input = LLMInput(20,500,args,10)
generate_res = llm_input.generate_time_series_without_dataloader(100,1,time_step_map, user_candidate_dict, load_data_user, load_item_user)
show_generate_res(generate_res)
# llm_input.reset_k(30)
# generate_res = llm_input.Generate(10)
# show_generate_res(generate_res)
# print("====Random Mode: Sample====")
# llm_input.set_random("sample")
# generate_res = llm_input.Generate(10)
# show_generate_res(generate_res)
# print("====Random Mode: Weighted====")
# llm_input.set_random("weighted")
# generate_res = llm_input.Generate(10)
# show_generate_res(generate_res)