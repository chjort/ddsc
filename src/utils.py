import json
import os


def save_as_json(obj: dict, path: str):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path) as f:
        data = json.load(f)

    return data
