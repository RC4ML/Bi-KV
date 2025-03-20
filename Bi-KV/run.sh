mpirun -np 5 --hostfile hostfile \
--oversubscribe \
--map-by node --rank-by slot \
-x MASTER_ADDR=10.0.0.2 \
-x MASTER_PORT=29051 \
-x PATH \
-x WORLD_SIZE=5 \
-x KVCACHE_NUM=2 \
-x WORKER_NUM=1 \
python3 distributed_init.py