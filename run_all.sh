#!/usr/bin/env bash
# Reproduce every number and figure in the README (CPU: about 3 hours on an
# Apple M2, most of it the version-5 comparison). Learning curves go to runs/
# (not committed), figures to static/.
set -e
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} MPLBACKEND=Agg
mkdir -p runs

python version1.py
python version2.py --task copy      --save_history runs/v2_copy_random.json
python version2.py --task selective --save_history runs/v2_selective_random.json --skip_demo
python version3.py --task selective --save_history runs/v3_selective.json
python version3.py --task copy      --save_history runs/v3_copy.json --skip_demo
python version4.py --task selective --save_history runs/v4_selective.json
python version4.py --task copy      --save_history runs/v4_copy.json --skip_demo

# version 5: demos once, then initialisations on selective copying (1000 steps)
python version5.py --skip_train
for cfg in "random default" "real loguniform"; do
  set -- $cfg
  python version5.py --task selective --A_init $1 --dt_init $2 --seed 0 --steps 1000 \
    --skip_demo --save_history runs/v5_selective_$1_$2_s0.json
done

# longer memory: 256 blanks (L = 264), one layer, d_model 32, batch 32
LONG="--task selective --n_mem 256 --n_layers 1 --d_model 32 --batch_size 32 --skip_demo"
python version2.py $LONG --steps 2000 --save_history runs/long_v2.json
python version5.py $LONG --steps 1500 --A_init random --dt_init default --save_history runs/long_v5_random_default.json
python version5.py $LONG --steps 1500 --A_init real --dt_init loguniform --save_history runs/long_v5_real_loguniform.json

python plot_results.py
