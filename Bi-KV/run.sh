mpirun -np 6 --hostfile hostfile \
--map-by node --rank-by slot \
-x MASTER_ADDR=10.0.0.1 \
-x MASTER_PORT=29051 \
-x PATH \
-x WORLD_SIZE=6 \
-x KVCACHE_NUM=2 \
-x WORKER_NUM=2 \
python3 distributed_init.py