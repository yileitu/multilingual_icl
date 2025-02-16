```shell
ssh yiletu@euler.ethz.ch
conda activate multilingual
my_share_info
scancel -u yiletu

du -sh cache
cd /cluster/project/sachan/model
python temp_download.py
cd /cluster/project/sachan/yilei/cache/hub
cd /cluster/home/yiletu

cd /cluster/project/sachan/yilei/projects/multiligual_examplar/data/GSM8K/

srun --mem-per-cpu=50G --gres=gpumem:23g --gpus=1 --time=4:00:00 --pty bash
srun --mem-per-cpu=30G --gpus=rtx_3090:1 --time=4:00:00 --pty bash
srun --mem-per-cpu=50G --gpus=a100_80gb:1 --time=4:00:00 --pty bash
srun --mem-per-cpu=50G --time=4:00:00 --pty bash
exit
chmod u+x euler_batch_run_mgsm_llm.sh
scancel -u yiletu
sbatch < euler_run_mgsm_chatgpt.sh

chmod u+x euler_batch_run_mgsm_llm.sh

cd /cluster/project/sachan/yilei/projects/multiligual_examplar/evaluation/MGSM
./euler_batch_run_mgsm_llm.sh
python3 eval_mgsm_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3.1 \
  --chat True \
  --eval_dataset mgsm \
  --n_shot 6 \
  --icl_mode japanese \
  --cot_mode english \
  --prepend_random_sentence True \
  --random_sentence_path data/FLORES/flores_10-15.json \
  --random_sentence_lang fr
 
python3 eval_mgsm_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type $model_type \
  --chat True \
  --eval_dataset mgsm \
  --n_shot 6 \
  --icl_mode $icl_mode \
  --cot_mode $cot_mode \
  --prepend_random_sentence True \
  --random_sentence_path data/FLORES/flores_10-15.json \
  --random_sentence_lang $random_sentence_lang
 
cd /cluster/project/sachan/yilei/projects/multiligual_examplar/evaluation/GSM8K
python3 eval_gsm8k.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --chat True \
  --eval_dataset gsm8k \
  --n_shot 6 \
  --icl_mode english \
  --cot_mode english
```

```shell
cd /cluster/project/sachan/yilei/projects/multiligual_examplar/evaluation/XLWIC

./euler_batch_run_xlwic_llm.sh
sbatch < euler_run_xlwic_chatgpt.sh

python3 eval_xlwic_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type aya \
  --chat True \
  --eval_dataset xlwic \
  --n_shot 6 \
  --icl_mode japanese \
  --cot_mode direct \
  --all_source_language True \
   --prepend_random_sentence True \
  --random_sentence_path data/FLORES/flores_10-15.json \
  --random_sentence_lang fr
 
python3 postprocess_model_response.py
```
```shell
cd /cluster/project/sachan/yilei/projects/multiligual_examplar/evaluation/XCOPA

./euler_batch_run_xcopa_llm.sh

python3 eval_xcopa_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type qwen2.5 \
  --chat True \
  --eval_dataset xcopa \
  --n_shot 6 \
  --icl_mode chinese \
  --cot_mode direct \
  --all_source_language True \
  --prepend_random_sentence True \
  --random_sentence_path data/FLORES/flores_10-15.json \
  --random_sentence_lang zh 
```
```shell
cd /cluster/project/sachan/yilei/projects/multiligual_examplar/evaluation/XNLI

./euler_batch_run_xnli_llm.sh

python3 eval_xnli_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --chat True \
  --eval_dataset xnli \
  --n_shot 8 \
  --icl_mode multilingual \
  --cot_mode direct \
  --all_source_language True
```
```shell
a100_80gb
a100-pcie-40gb
v100
quadro_rtx_6000
rtx_4090
rtx_3090
titan_rtx
rtx_2080_ti
```
```shell
cd /cluster/project/sachan/yilei/projects/multiligual_examplar/neuron

./euler_run_record_act.sh
python3 record_act.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset mgsm \
  --n_shot 6 \
  --cot_mode english
python3 record_act.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3.1 \
  --eval_dataset xlwic \
  --n_shot 6 \
  --cot_mode direct
python3 record_act.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset xcopa \
  --n_shot 5 \
  --cot_mode direct
python3 record_act.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset xnli \
  --n_shot 8 \
  --cot_mode direct
python3 record_act.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset gsm8k \
  --n_shot 6 \
  --cot_mode english

./euler_run_identify.sh
python3 identify_specific_neurons.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset xlwic \
  --act_percentile 0.7
  
python3 identify_specific_neurons.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset mgsm \
  --act_progressive_threshold 0.1
 
./euler_run_deact.sh
python3 deactivate.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --eval_dataset xlwic \
  --n_shot 6 \
  --cot_mode direct \
  --deact_mode paired \
  --neuron_path act_over_zero_cnt/xlwic/llama3-8b-instruct/percentile_0.7
```
```shell
cd /cluster/project/sachan/yilei/projects/multiligual_examplar/backprop

./euler_batch_run_grad_norm.sh

python3 grad_norm.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type qwen2 \
  --chat True \
  --eval_dataset xlwic \
  --n_shot 6 \
  --icl_mode english \
  --cot_mode direct \
  --all_source_language False
```

```shell
ssh yileitu@vremote.vectorinstitute.ai
ssh yileitu@vlogin.vectorinstitute.ai
ssh yileitu@vdm1.vectorinstitute.ai
ssh yileitu@v.vectorinstitute.ai

cat ~/.usage

cd /h/yileitu
cd /scratch/ssd004/scratch/yileitu/hf_cache
cd /h/yileitu/hf_cache/hub/models--mistralai--Mistral-Nemo-Instruct-2407
cd /datasets
cd /model-weights

nano ~/.bashrc
source ~/.bashrc

module load cuda-12.1
>>> print(torch.__version__)
2.4.1+cu121

srun -c 1 --gres=gpu:a40:1 --mem=30GB --pty --time=4:00:00 bash
conda activate multilingual
squeue -u yileitu

sinfo -N -l -p a40

t4v1
t4v2
rtx6000
a40
a100


/model-weights/Meta-Llama-3-8B-Instruct
/model-weights/Meta-Llama-3.1-8B-Instruct
/model-weights/Qwen2-7B-Instruct
/model-weights/Mistral-7B-Instruct-v0.3


python eval_mgsm_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3.1 \
  --model_hf_path /model-weights/Meta-Llama-3.1-8B-Instruct \
  --eval_dataset mgsm \
  --n_shot 6 \
  --icl_mode english \
  --cot_mode english \
  --prepend_random_sentence False \
  --random_sentence_path data/FLORES/flores_10-15_all_high_langs.json \
  --random_sentence_lang multilingual \
  --google_translate_test_questions False \
  --google_translate_demonstrations True
  
python eval_mgsm_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type aya \
  --eval_dataset mgsm \
  --n_shot 6 \
  --icl_mode english \
  --cot_mode english \
  --prepend_random_sentence False \
  --random_sentence_path data/FLORES/flores_10-15_all_high_langs.json \
  --random_sentence_lang multilingual
  
  
python3 eval_xcopa_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3.1 \
  --model_hf_path /model-weights/Meta-Llama-3.1-8B-Instruct \
  --chat True \
  --eval_dataset xcopa \
  --n_shot 6 \
  --icl_mode english \
  --cot_mode direct \
  --all_source_language True \
  --prepend_random_sentence False \
  --random_sentence_path data/FLORES/flores_10-15_all_high_langs.json \
  --random_sentence_lang multilingual \
  --google_translate_test_questions False \
  --google_translate_demonstrations True
  
  
python3 eval_xlwic_chat_template.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type mistral-nemo \
  --chat True \
  --eval_dataset xlwic \
  --n_shot 6 \
  --icl_mode english \
  --cot_mode direct \
  --all_source_language True \
  --prepend_random_sentence False \
  --random_sentence_path data/FLORES/flores_10-15_all_high_langs.json \
  --random_sentence_lang multilingual


cd /h/yileitu/multilingual_exemplar/evaluation/MGSM
./vector_batch_run_mgsm_llm.sh
./vector_run_mgsm_chatgpt.sh

cd /h/yileitu/multilingual_exemplar/evaluation/XCOPA
./vector_batch_run_xcopa_llm.sh

cd /h/yileitu/multilingual_exemplar/evaluation/XLWIC
./vector_batch_run_xlwic_llm.sh

conda activate multilingual
cd /h/yileitu/multilingual_exemplar/backprop

./vector_batch_run_grad_norm.sh
python3 grad_norm.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3.1 \
  --chat True \
  --model_hf_path /model-weights/Meta-Llama-3.1-8B-Instruct \
  --eval_dataset xcopa \
  --n_shot 5 \
  --icl_mode native \
  --cot_mode direct

./vector_batch_run_token_removal.sh
python3 remove_token_by_grad_percentile.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3.1 \
  --chat True \
  --model_hf_path /model-weights/Meta-Llama-3.1-8B-Instruct \
  --eval_dataset xcopa \
  --n_shot 5 \
  --icl_mode english \
  --cot_mode direct \
  --token_removal_percentage 0.3

./vector_batch_run_token_lang_pct.sh
python3 token_lang_pct.py \
  --seed 21946520 \
  --n_gpu 1 \
  --model_type llama3 \
  --chat True \
  --model_hf_path /model-weights/Meta-Llama-3.1-8B-Instruct \
  --eval_dataset mgsm \
  --n_shot 6 \
  --icl_mode multilingual \
  --cot_mode direct \
  --all_source_language True \
  --token_removal_percentage 0.3


cd /h/yileitu/multilingual_exemplar/evaluation/postprocessing
python3 combine_all_eval_acc.py
```

```shell
cd /h/yileitu/multilingual_exemplar/evaluation/MGSM
chmod u+x vector_batch_run_mgsm_llm.sh
./vector_batch_run_mgsm_llm.sh
./vector_run_mgsm_chatgpt.sh

cd /h/yileitu/multilingual_exemplar/evaluation/XCOPA
./vector_batch_run_xcopa_llm.sh

cd /h/yileitu/multilingual_exemplar/evaluation/XLWIC
./vector_batch_run_xlwic_llm.sh
```

```
gpt-3.5-turbo-0125
gpt-4-turbo-2024-04-09
gpt-4o-mini
```

```shell
conda activate multilingual
cd /h/yileitu/multilingual_exemplar/evaluation/postprocessing
python3 combine_all_eval_acc.py

cd /h/yileitu/multilingual_exemplar/hypothesis_test/
./vector_run_hyp_test.sh

python3 hyp_test.py \
  --dataset XCOPA \
  --test_method mcnemar \
  --test_case 1

multi_icl+NA noise vs en_icl+multi noise
multi_icl+en noise vs en_icl+multi noise

zip -r multilingual_eval.zip MGSM XCOPA XLWIC
tar -zcvf multilingual_eval.tar.gz MGSM XCOPA XLWIC


gpt-3.5-turbo-0125/icl-native_cot-english/mgsm_evaluation_results_te.json
gpt-4o-mini-2024-07-18/icl-native_cot-english/mgsm_evaluation_results_te.json
```

