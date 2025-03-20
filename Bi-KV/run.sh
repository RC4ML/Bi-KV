mpirun -np 8 --hostfile hostfile \
--oversubscribe \
--map-by node --rank-by slot \
-x MASTER_ADDR=192.168.189.8 \
-x MASTER_PORT=7471 \
-x PATH \
-x WORLD_SIZE=8 \
-x KVCACHE_NUM=3 \
-x WORKER_NUM=3 \
python3 distributed_grpc_init.py