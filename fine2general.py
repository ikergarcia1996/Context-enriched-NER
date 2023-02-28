from typing import Dict
import os

fine2general: Dict[str, str] = {
    "Facility": "Location",
    "OtherLOC": "Location",
    "HumanSettlement": "Location",
    "Station": "Location",
    "VisualWork": "CreativeWork",
    "MusicalWork": "CreativeWork",
    "WrittenWork": "CreativeWork",
    "ArtWork": "CreativeWork",
    "Software": "CreativeWork",
    "OtherCW": "CreativeWork",
    "MusicalGRP": "Group",
    "PublicCorp": "Group",
    "PrivateCorp": "Group",
    "OtherGRP": "Group",
    "AerospaceManufacturer": "Group",
    "SportsGRP": "Group",
    "CarManufacturer": "Group",
    "TechCorp": "Group",
    "ORG": "Group",
    "Scientist": "Person",
    "Artist": "Person",
    "Athlete": "Person",
    "Politician": "Person",
    "Cleric": "Person",
    "SportsManager": "Person",
    "OtherPER": "Person",
    "Clothing": "Product",
    "Vehicle": "Product",
    "Food": "Product",
    "Drink": "Product",
    "OtherPROD": "Product",
    "Medication/Vaccine": "Medical",
    "MedicalProcedure": "Medical",
    "AnatomicalStructure": "Medical",
    "Symptom": "Medical",
    "Disease": "Medical",
}


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
                    if cat in fine2general:
                        print(f"{word} {pos}-{fine2general[cat]}", file=output_file)
                    else:
                        raise ValueError(f"Unknown category: {cat}")


if __name__ == "__main__":
    os.makedirs(
        "/ikerlariak/igarcia945/MultiCoNER2/MultiCoNER2_train_dev_test/general",
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
                dataset_path=f"/ikerlariak/igarcia945/MultiCoNER2/MultiCoNER2_train_dev_test/finegrained/{lang}_{split}.conll",
                output_path=f"/ikerlariak/igarcia945/MultiCoNER2/MultiCoNER2_train_dev_test/general/{lang}_{split}.conll",
            )

