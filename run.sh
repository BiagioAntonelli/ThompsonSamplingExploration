#!/bin/bash

#python run_simulation.py --exp 1 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent vanillaNN
#python run_simulation.py --exp 2 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent dropoutNN
#python run_simulation.py --exp 3 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianNN
#python run_simulation.py --exp 4 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianLR-1
#python run_simulation.py --exp 5 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianLR_exploit

#python run_simulation.py --exp 6 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent vanillaNN --deep_gt True
#python run_simulation.py --exp 7 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent dropoutNN --deep_gt True
#python run_simulation.py --exp 8 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianNN --deep_gt True
#python run_simulation.py --exp 9 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianLR-1 --deep_gt True
#python run_simulation.py --exp 10 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianLR_exploit --deep_gt True

#python run_simulation.py --exp 11 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianLR-05
#python run_simulation.py --exp 12 --freq 10 --n_train 100  --len_sim 2000 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianLR-05 --deep_gt True

python run_simulation.py --exp 0 --freq 10 --n_train 100  --len_sim 300 --n_ads_sel 100 --n_new_ads 1000 --agent bayesianNN --deep_gt True