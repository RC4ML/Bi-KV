# Bi-KV
Bipartite KVCache

## Requirements
```
pip install transformers
pip install sentencepiece
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install vllm
```
## Run
```
cd Bi-KV
KVCACHE_NUM=5 WORKER_NUM=5 python init.py  # set number of kvcache and worker 
```
## Distributed Run
```
cd Bi-KV
python distributed_run.py
```
