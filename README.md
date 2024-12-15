# Bi-KV
Bipartite KVCache

## Requirements
```
pip install transformers
pip install sentencepiece
```
## Run
```
cd Bi-KV
KVCACHE_NUM=5 WORKER_NUM=5 python init.py  # set number of kvcache and worker 
```
## Distributed Run
```
cd Bi-KV
bash run.sh
```
