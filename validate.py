import argparse
import json
from pathlib import Path
from aml.validate import validate_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all required output artifacts")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    print(json.dumps(validate_outputs(args.data, args.out), ensure_ascii=False, indent=2))
