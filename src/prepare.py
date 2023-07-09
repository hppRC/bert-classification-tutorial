import random
import unicodedata
from pathlib import Path

from more_itertools import divide, flatten
from tap import Tap
from tqdm import tqdm

import src.utils as utils


class Args(Tap):
    input_dir: Path = "./data/text"
    output_dir: Path = "./datasets/livedoor"
    seed: int = 42


def process_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = title.strip("　").strip()
    return title


# 記事本文の前処理
# 重複した改行の削除、文頭の全角スペースの削除、NFKC正規化を実施
def process_body(body: list[str]) -> str:
    body = [unicodedata.normalize("NFKC", line) for line in body]
    body = [line.strip("　").strip() for line in body]
    body = [line for line in body if line]
    body = "\n".join(body)
    return body


def main(args: Args):
    random.seed(args.seed)

    data = []
    labels = set()

    for path in tqdm(list(args.input_dir.glob("*/*.txt"))):
        if path.name == "LICENSE.txt":
            continue
        category = path.parent.name
        labels.add(category)

        # データフォーマット
        # １行目：記事のURL
        # ２行目：記事の日付
        # ３行目：記事のタイトル
        # ４行目以降：記事の本文
        lines = path.read_text().splitlines()

        data.append(
            {
                "category": category,
                "category-id": path.stem,
                "url": lines[0].strip(),
                "date": lines[1].strip(),
                "title": process_title(lines[2].strip()),
                "body": process_body(lines[3:]),
            }
        )
    random.shuffle(data)

    label2id = {label: i for i, label in enumerate(sorted(labels))}
    data = [
        {
            "id": idx,
            "label": label2id[d["category"]],
            **d,
        }
        for idx, d in enumerate(data)
    ]

    utils.save_jsonl(data, args.output_dir / "all.jsonl")
    utils.save_json(label2id, args.output_dir / "label2id.json")

    portions = list(divide(10, data))
    train, val, test = list(flatten(portions[:-2])), portions[-2], portions[-1]
    utils.save_jsonl(train, args.output_dir / "train.jsonl")
    utils.save_jsonl(val, args.output_dir / "val.jsonl")
    utils.save_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
