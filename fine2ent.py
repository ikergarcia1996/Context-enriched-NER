import os


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
    os.makedirs(
        "/ikerlariak/igarcia945/MultiCoNER2/MultiCoNER2_train_dev_test/entity/",
        exist_ok=True,
    )
    for lang in [
        "bn",
        "de",
        "en",
        "es",
        "fa",
        "fr",
        "hi",
        "it",
        "pt",
        "sv",
        "uk",
        "zh",
        "multi"
    ]:
        for split in ["train", "dev","test"]:
            dataset2general(
                dataset_path=f"multiconer2023/finegrained/{lang}_{split}.conll",
                output_path=f"multiconer2023/entity/{lang}_{split}.conll",
            )

