from eval_utils import evaluate_tsv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predictions",
    type=str,
    help="Predictions in tsv format",
)

parser.add_argument(
    "--gold",
    type=str,
    help="Gold in tsv format",
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Output dir",
)

args = parser.parse_args()

evaluate_tsv(
    original_dataset_path=args.gold,
    preds_path=args.predictions,
    output_dir=args.output_dir,
    output_name=os.path.splitext(os.path.basename(args.predictions))[0],
    set_unique_label=False,
)

# python3 evaluate_predictions.py --predictions /ikerlariak/igarcia945/Seq2SeqT5/GENRE/results/genre_wiki/pipeline/en/val_logs/epoch_1.tsv --gold /ikerlariak/igarcia945/MultiCoNER2/MultiCoNER_2_train_dev/train_dev_clean/en-dev.conll --output_dir /ikerlariak/igarcia945/Seq2SeqT5/GENRE/results/genre_wiki/pipeline/en
