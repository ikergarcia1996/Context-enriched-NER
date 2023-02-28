#!/bin/bash
#SBATCH --job-name=genre
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=genre.out.txt
#SBATCH --error=genre.err.txt

source /ikerlariak/igarcia945/envs/pytorch-tximista/bin/activate

cd ..

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"

for lang in bn de en es fa fr hi it pt sv uk zh multi
do
# For the train and development set, we use the gold label boundaries
python3 run_genre.py \
--tsv_path ./multiconer2023/entity/"$lang"_train.conll \
--batch_size 32 \
--output_path ./results/genre/"$lang"/"$lang"_train.json

python3 run_genre.py \
--tsv_path ./multiconer2023/entity/"$lang"_dev.conll \
--batch_size 32 \
--output_path ./results/genre/"$lang"/"$lang"_dev.json

# For the test set, we use the predicted boundaries
python3 run_genre.py \
--tsv_path ../results/entity_boundaries/"$lang"/test.model_predictions.tsv \
--batch_size 32 \
--output_path ./results/genre/"$lang"/"$lang"_test.json

done