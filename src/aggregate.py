from pathlib import Path

import pandas as pd
from classopt import classopt

import src.utils as utils


@classopt(default_long=True)
class Args:
    output_dir: Path = "./outputs"


def main(args: Args):
    data = []
    for path in args.output_dir.glob("**/test-metrics.json"):
        metrics = utils.load_json(path)

        config = utils.load_json(path.parent / "config.json")
        model_name = config["model_name"]

        data.append(
            {
                "model_name": model_name,
                **metrics,
            }
        )

    data = sorted(data, key=lambda d: d["f1"], reverse=True)
    pd.DataFrame(data).to_csv("metrics.csv", index=False)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
