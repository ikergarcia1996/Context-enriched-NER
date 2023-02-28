#!/bin/bash
#SBATCH --job-name=genre
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
python3 get_wikidata.py \
--json_dict_path ./results/genre/"$lang"/"$lang"_train.json  \
--language en \
--batch_size 256 \
--num_workers 32 \
--ignore_cache

python3 get_wikidata.py \
--json_dict_path ./results/genre/"$lang"/"$lang"_dev.json  \
--language en \
--batch_size 256 \
--num_workers 32 \
--ignore_cache

python3 get_wikidata.py \
--json_dict_path ./results/genre/"$lang"/"$lang"_test.json  \
--language en \
--batch_size 256 \
--num_workers 32 \
--ignore_cache

done