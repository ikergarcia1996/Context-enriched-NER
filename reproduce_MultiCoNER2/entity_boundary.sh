#!/bin/bash
#SBATCH --job-name=entity_boundary
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=entity_boundary.out.txt
#SBATCH --error=entity_boundary.err.txt

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"


cd ../TokenClassification/ || exit


for lang in bn de en es fa fr hi it pt sv uk zh multi
do

python3 run_tokenclass.py \
--train_file ./multiconer2023/entity/"$lang"_train.conll \
--dev_file ./multiconer2023/entity/"$lang"_dev.conll \
--test_file ./multiconer2023/entity/"$lang"_test.conll \
--model_name xlm-roberta-large \
--output_dir ./results/entity_boundaries/"$lang" \
--num_train_epochs 8 \
--batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--max_seq_length 256 \
--number_of_experiments 5 \
--experiment_name ./results/entity_boundaries/"$lang" \
--encoding iob2 \
--lr_scheduler_type cosine \
--fp16

done