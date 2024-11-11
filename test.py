from config import *
from datasets import dataset_factory
from dataloader import LLMDataloader
from inputGenerator.inputGenerator import LLMInput
args.model_code = 'llm'
args.llm_retrieved_path = "/data/testmodel/LlamaRec/experiments/lru/games/"
args.dataset_code = "games"
set_template(args)
dataset = dataset_factory(args)
dataloader = LLMDataloader(args, dataset)
llmDataset = dataloader._get_eval_dataset()

llm_input = LLMInput(10,llmDataset)
generate_res = llm_input.Generate(10)
for i in generate_res:
    print(i)