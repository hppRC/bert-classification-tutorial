from pathlib import Path

import pandas as pd
import torch.nn as nn
from tap import Tap
from transformers import AutoModel

import src.utils as utils


class Args(Tap):
    input_dir: Path = "./outputs"
    output_dir: Path = "./results"


def calc_num_params(model: nn.Module) -> int:
    params = 0
    for p in model.parameters():
        params += p.numel()
    return params


def main(args: Args):
    data = []
    for path in args.input_dir.glob("**/test-metrics.json"):
        test_metrics = utils.load_json(path)
        val_metrics = utils.load_json(path.parent / "val-metrics.json")
        config = utils.load_json(path.parent / "config.json")

        data.append(
            {
                "model_name": config["model_name"],
                "lr": config["lr"],
                "best-val-epoch": val_metrics["best-epoch"],
                "best-val-f1": val_metrics["f1"],
                **test_metrics,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data).sort_values("f1", ascending=False)
    df.to_csv(str(args.output_dir / "all.csv"), index=False)

    best_df = (
        df.groupby("model_name", as_index=False)
        .apply(lambda x: x.nlargest(1, "best-val-f1").reset_index(drop=True))
        .reset_index(drop=True)
    ).sort_values("f1", ascending=False)

    best_df.to_csv(str(args.output_dir / "best.csv"), index=False)

    for row in best_df.to_dict("records"):
        model = AutoModel.from_pretrained(row["model_name"])
        params = calc_num_params(model) // 1_000_000

        print(
            f'|[{row["model_name"]}](https://huggingface.co/{row["model_name"]})|{params} M|{row["lr"]}|{row["accuracy"]*100:.2f}|{row["precision"]*100:.2f}|{row["recall"]*100:.2f}|{row["f1"]*100:.2f}|'
        )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
