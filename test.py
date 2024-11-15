from config import *
from datasets import dataset_factory
from dataloader import LLMDataloader
from inputGenerator.inputGenerator import LLMInput
# from huggingface_hub import login
# login()
args.model_code = 'llm'
args.llm_retrieved_path = "/share/gnn_data/testmodel/LlamaRec/experiments/lru/games/"
args.dataset_code = "games"
set_template(args)

llm_input = LLMInput(20,500,args)
generate_res = llm_input.Generate(100)
for ind,i in enumerate(generate_res):
    print(f"User ID: {i['user_id']}\nUser History Tokens: {i['user_history_tokens']}\nItems: {i['items']}\nTimestamp: {i['timestamp']}\nItem Count:{len(i['items'])}")
print("====Change k to 30====")
llm_input.reset_k(30)
generate_res = llm_input.Generate(10)
for ind,i in enumerate(generate_res):
    print(f"User ID: {i['user_id']}\nUser History Tokens: {i['user_history_tokens']}\nItems: {i['items']}\nTimestamp: {i['timestamp']}\nItem Count:{len(i['items'])}")