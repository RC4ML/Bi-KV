mpirun -np 2 --hostfile hostfile \
--oversubscribe \
--map-by node --rank-by slot \
-x WORLD_SIZE=2 \
python3 benchallreduce.py