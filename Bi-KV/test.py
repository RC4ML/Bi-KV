from typing import List
from config import *
from datasets import dataset_factory
from dataloader import LLMDataloader
from inputGenerator.inputGenerator import LLMInput,InputPrompt
# from huggingface_hub import login
# login()
args.model_code = 'llm'
args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
# args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games"
args.dataset_code = "games"
set_template(args)

def show_generate_res(res_list:List[InputPrompt]):
    for i in res_list:
        print(f"User ID: {i.user_id}\n\
User History Tokens: {i.user_history_tokens}\n\
Items: {i.items}\n\
Timestamp: {i.timestamp}\n\
Item Count: {len(i.items)}")

llm_input = LLMInput(20,500,args)
generate_res = llm_input.generate(100)
show_generate_res(generate_res)
llm_input.reset_k(30)
generate_res = llm_input.generate(10)
show_generate_res(generate_res)
print("====Random Mode: Sample====")
llm_input.set_random("sample")
generate_res = llm_input.generate(10)
show_generate_res(generate_res)
print("====Random Mode: Weighted====")
llm_input.set_random("weighted")
generate_res = llm_input.generate(10)
show_generate_res(generate_res)