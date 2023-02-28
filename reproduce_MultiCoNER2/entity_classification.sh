#!/bin/bash
#SBATCH --job-name=entityclassification
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=entityclassification.out.txt
#SBATCH --error=entityclassification.err.txt

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ..

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"

for lang in bn de en es fa fr hi it pt sv uk zh multi
do

accelerate launch --mixed_precision fp16  run_text_classification.py \
--train_json_path ./results/genre/"$lang"/"$lang"_train.json \
--dev_json_path ./results/genre/"$lang"/"$lang"_dev.json \
--test_json_path ./results/genre/"$lang"/"$lang"_test.json \
--model_name_or_path xlm-roberta-large \
--output_dir ./results/finegrained/"$lang" \
--label_category fine \
--max_len 256 \
--train_batch_size 16 \
--gradient_accumulation_steps 1 \
--eval_batch_size 64 \
--learning_rate 2e-5  \
--num_train_epochs 8 \
--do_not_save_model \
--include_wikidata_description \
--include_wikidata_arguments \
--include_wikipedia_summary \
--number_of_experiments 5 \

done