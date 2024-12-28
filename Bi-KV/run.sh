mpirun -np 4 --hostfile hostfile \
--map-by node --rank-by slot \
-x MASTER_ADDR=10.0.0.2 \
-x MASTER_PORT=29051 \
-x PATH \
-x WORLD_SIZE=4 \
-x KVCACHE_NUM=1 \
-x WORKER_NUM=1 \
python3 distributed_init.py