#!/bin/bash

#declare -a datasets=("XLWIC" "XCOPA")
declare -a datasets=("MGSM")
declare -a test_methods=("mcnemar")
declare -a test_cases=(2)

for dataset in "${datasets[@]}"; do
  for test_method in "${test_methods[@]}"; do
    for test_case in "${test_cases[@]}"; do
      sbatch <<EOT
#!/bin/bash -l

#SBATCH --time=04:00:00
#SBATCH --qos=m3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:rtx6000:1

conda activate multilingual

python3 hyp_test.py \
  --dataset $dataset \
  --test_method $test_method \
  --test_case $test_case
EOT
    done
  done
done
