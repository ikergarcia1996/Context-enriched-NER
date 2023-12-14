import os
import argparse


def dataset2general(dataset_path: str, output_path: str):
    with open(dataset_path, "r") as dataset_file, open(output_path, "w") as output_file:
        for line in dataset_file:
            line = line.strip()
            if line == "":
                print(file=output_file)
            else:
                word, label = line.split()
                if label == "O":
                    print(f"{word} {label}", file=output_file)
                else:
                    pos, cat = label.split("-")
                    print(f"{word} {pos}-ENTITY", file=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_dataset), exist_ok=True)
    dataset2general(args.input_dataset, args.output_dataset)
