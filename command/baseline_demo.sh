#!/bin/bash
#SBATCH -c 8               
#SBATCH -t 0-24:00           
#SBATCH -p kempner   
#SBATCH --gres=gpu:1
#SBATCH --mem=10000            
#SBATCH -o nn_model_%j.out   
#SBATCH -e nn_model_%j.err 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

for data in "000016" "000300" "000852" "000903" "000905"; do
    python src/run_baseline.py --data $data
done
