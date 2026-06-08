import json
from pathlib import Path

MFH_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = MFH_DIR / "mfh_buildings_combined.json"


def load_buildings():
    buildings = []
    for fp in sorted(MFH_DIR.glob("*.json")):
        if fp.name == OUTPUT_PATH.name:
            continue
        with open(fp, encoding="utf-8") as f:
            buildings.append(json.load(f))
    buildings.sort(key=lambda b: b.get("building", {}).get("id", ""))
    return buildings


def build_combined(buildings):
    return {
        "metadata": {
            "source_directory": "data-json-TABULA/multifamlyhouse",
            "building_type": "multifamily_house",
            "size_class": "MFH",
            "building_count": len(buildings),
            "source_files": [
                b["building"]["building_json"]
                for b in buildings
                if "building" in b and "building_json" in b["building"]
            ],
        },
        "buildings": buildings,
    }


def main():
    buildings = load_buildings()
    combined = build_combined(buildings)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"Created: {OUTPUT_PATH.name}")
    print(f"Buildings merged: {len(buildings)}")
    for building in buildings:
        print(f"  - {building['building']['id']}")


if __name__ == "__main__":
    main()
