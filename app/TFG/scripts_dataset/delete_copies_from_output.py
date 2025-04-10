import json
import argparse
from pathlib import Path

def main(args):
    input_path = Path(args.dataset_path)
    output_path = input_path.with_name(input_path.stem + "_deduplicated.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["ground_truths", "predictions"]:
        if key in data:
            seen = set()
            unique = []
            for item in data[key]:
                item_str = json.dumps(item, sort_keys=True)
                if item_str not in seen:
                    seen.add(item_str)
                    unique.append(item)
            data[key] = unique

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Deduplicated file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, default="./final_dataset_fatura/test",
        help="Local path from ./app to the dataset."
    )

    args, left_argv = parser.parse_known_args()
    main(args)
