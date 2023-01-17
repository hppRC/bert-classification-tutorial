from datetime import datetime
from pathlib import Path

import torch
from classopt import classopt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

import src.utils as utils


@classopt(default_long=True)
class Args:
    model_name: str = "cl-tohoku/bert-base-japanese-v2"
    dataset_dir: Path = "./datasets/livedoor"

    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-5
    num_warmup_epochs: int = 2
    max_seq_len: int = 512

    date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    device: str = "cuda:0"
    seed: int = 42

    def __post_init__(self):
        utils.set_seed(self.seed)

        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("outputs") / model_name / self.date
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> list[dict]:
    return utils.load_jsonl(path).to_dict(orient="records")


def main(args: Args):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_seq_len,
    )
    model: PreTrainedModel = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(args.labels),
        )
        .eval()
        .to(args.device, non_blocking=True)
    )

    train_dataset: list[dict] = load_dataset(args.dataset_dir / "train.jsonl")
    val_dataset: list[dict] = load_dataset(args.dataset_dir / "val.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test.jsonl")

    def collate_fn(data_list: list[dict]) -> BatchEncoding:
        title = [d["title"] for d in data_list]
        body = [d["body"] for d in data_list]

        inputs: BatchEncoding = tokenizer(
            title,
            body,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

        labels = torch.LongTensor([d["label"] for d in data_list])
        return BatchEncoding({**inputs, "labels": labels})

    def create_loader(dataset, batch_size=None, shuffle=False):
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    train_dataloader: DataLoader = create_loader(train_dataset, shuffle=True)
    val_dataloader: DataLoader = create_loader(val_dataset, shuffle=False)
    test_dataloader: DataLoader = create_loader(test_dataset, shuffle=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * args.num_warmup_epochs,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    def clone_state_dict() -> dict:
        return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    @torch.inference_mode()
    def evaluate(dataloader: DataLoader) -> dict[str, float]:
        model.eval()
        loss = 0
        gold_labels, pred_labels = [], []

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            out: SequenceClassifierOutput = model(**batch.to(args.device))
            batch_size: int = batch.input_ids.size(0)
            loss += out.loss.item() * batch_size

            gold_labels += batch.labels.tolist()
            pred_labels += out.logits.argmax(dim=-1).tolist()

        accuracy: float = accuracy_score(gold_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=args.labels,
        )

        return {
            "loss": loss / len(dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def log(metrics: dict) -> None:
        utils.log(metrics, args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"loss: {metrics['loss']:2.6f}   \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"precision: {metrics['precision']:.4f} \t"
            f"recall: {metrics['recall']:.4f} \t"
            f"f1: {metrics['f1']:.4f}"
        )

    val_metrics = {"epoch": None, **evaluate(val_dataloader)}
    best_epoch, best_val_f1 = None, val_metrics["f1"]
    best_state_dict = clone_state_dict()
    log(val_metrics)

    for epoch in trange(args.epochs, dynamic_ncols=True):
        model.train()

        for batch in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            dynamic_ncols=True,
            leave=False,
        ):
            out: SequenceClassifierOutput = model(**batch.to(args.device))
            loss: torch.FloatTensor = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        model.eval()
        val_metrics = {"epoch": epoch, **evaluate(val_dataloader)}
        log(val_metrics)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state_dict = clone_state_dict()

    model.load_state_dict(best_state_dict)
    model.eval().to(args.device, non_blocking=True)

    val_metrics = {"best-epoch": best_epoch, **evaluate(val_dataloader)}
    utils.save_json(val_metrics, args.output_dir / "val-metrics.json")

    test_metrics = evaluate(test_dataloader)
    utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
