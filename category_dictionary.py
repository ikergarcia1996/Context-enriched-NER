from typing import Dict, List

general_categories: List[str] = [
    "Location",
    "Person",
    "Group",
    "CreativeWork",
    "Product",
    "Medical",
]
general_category2id: Dict[str, int] = {
    category: id for id, category in enumerate(general_categories)
}
id2general_category: Dict[int, str] = {
    id: category for category, id in general_category2id.items()
}


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

general2fine: Dict[str, List[str]] = {
    "Location": [
        "Facility",
        "HumanSettlement",
        "Station",
        "OtherLOC",
    ],
    "CreativeWork": [
        "VisualWork",
        "MusicalWork",
        "WrittenWork",
        "ArtWork",
        "Software",
        "OtherCW",
    ],
    "Group": [
        "MusicalGRP",
        "PublicCorp",
        "PrivateCorp",
        "AerospaceManufacturer",
        "SportsGRP",
        "CarManufacturer",
        "TechCorp",
        "ORG",
        "OtherGRP",
    ],
    "Person": [
        "Scientist",
        "Artist",
        "Athlete",
        "Politician",
        "Cleric",
        "SportsManager",
        "OtherPER",
    ],
    "Product": [
        "Clothing",
        "Vehicle",
        "Food",
        "Drink",
        "OtherPROD",
    ],
    "Medical": [
        "Medication/Vaccine",
        "MedicalProcedure",
        "AnatomicalStructure",
        "Symptom",
        "Disease",
    ],
}

fine2id: Dict[str, int] = {
    "Facility": 0,
    "OtherLOC": 1,
    "HumanSettlement": 2,
    "Station": 3,
    "VisualWork": 4,
    "MusicalWork": 5,
    "WrittenWork": 6,
    "ArtWork": 7,
    "Software": 8,
    "OtherCW": 9,
    "MusicalGRP": 10,
    "PublicCorp": 11,
    "PrivateCorp": 12,
    "OtherGRP": 13,
    "AerospaceManufacturer": 14,
    "SportsGRP": 15,
    "CarManufacturer": 16,
    "TechCorp": 17,
    "ORG": 18,
    "Scientist": 19,
    "Artist": 20,
    "Athlete": 21,
    "Politician": 22,
    "Cleric": 23,
    "SportsManager": 24,
    "OtherPER": 25,
    "Clothing": 26,
    "Vehicle": 27,
    "Food": 28,
    "Drink": 29,
    "OtherPROD": 30,
    "Medication/Vaccine": 31,
    "MedicalProcedure": 32,
    "AnatomicalStructure": 33,
    "Symptom": 34,
    "Disease": 35,
}

id2fine: Dict[int, str] = {x: y for y, x in fine2id.items()}


