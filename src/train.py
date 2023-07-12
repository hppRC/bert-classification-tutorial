from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tap import Tap
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


class Args(Tap):
    model_name: str = "cl-tohoku/bert-base-japanese-v3"
    dataset_dir: Path = "./datasets/livedoor"

    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-5
    num_warmup_epochs: int = 2
    max_seq_len: int = 512
    weight_decay: float = 0.01
    gradient_checkpointing: bool = False

    device: str = "cuda:0"
    seed: int = 42

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())

        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
        self.output_dir = self.make_output_dir(
            "outputs",
            self.model_name,
            date,
            time,
        )

    def make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class Experiment:
    def __init__(self, args: Args):
        self.args: Args = args

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=args.max_seq_len,
        )

        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=len(args.labels),
            )
            .eval()
            .to(args.device, non_blocking=True)
        )

        # gradient_checkpointingとtorch.compileは相性が悪いことが多いので排他的に使用
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        else:
            self.model = torch.compile(self.model)

        self.train_dataloader: DataLoader = self.load_dataset(split="train", shuffle=True)
        self.val_dataloader: DataLoader = self.load_dataset(split="val")
        self.test_dataloader: DataLoader = self.load_dataset(split="test")

        self.optimizer, self.lr_scheduler = self.create_optimizer()

    def load_dataset(
        self,
        split: str,
        shuffle: bool = False,
    ) -> DataLoader:
        path: Path = self.args.dataset_dir / f"{split}.jsonl"
        dataset: list[dict] = utils.load_jsonl(path).to_dict(orient="records")
        return self.create_loader(dataset, shuffle=shuffle)

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        title = [d["title"] for d in data_list]
        body = [d["body"] for d in data_list]

        inputs: BatchEncoding = self.tokenizer(
            title,
            body,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

        labels = torch.LongTensor([d["label"] for d in data_list])
        return BatchEncoding({**inputs, "labels": labels})

    def create_loader(
        self,
        dataset,
        batch_size=None,
        shuffle=False,
    ):
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    def create_optimizer(self) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        # see: https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    param for name, param in self.model.named_parameters() if not name in no_decay
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param for name, param in self.model.named_parameters() if name in no_decay
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(self.train_dataloader) * args.num_warmup_epochs,
            num_training_steps=len(self.train_dataloader) * args.epochs,
        )

        return optimizer, lr_scheduler

    @torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None)
    def run(self):
        val_metrics = {"epoch": None, **self.evaluate(self.val_dataloader)}
        best_epoch, best_val_f1 = None, val_metrics["f1"]
        best_state_dict = self.clone_state_dict()
        self.log(val_metrics)

        scaler = torch.cuda.amp.GradScaler()

        for epoch in trange(args.epochs, dynamic_ncols=True):
            self.model.train()

            for batch in tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                dynamic_ncols=True,
                leave=False,
            ):
                out: SequenceClassifierOutput = self.model(**batch.to(args.device))
                loss: torch.FloatTensor = out.loss

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)

                scale = scaler.get_scale()
                scaler.update()
                if scale <= scaler.get_scale():
                    self.lr_scheduler.step()

            self.model.eval()
            val_metrics = {"epoch": epoch, **self.evaluate(self.val_dataloader)}
            self.log(val_metrics)

            # 開発セットでのF値最良時のモデルを保存
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_state_dict = self.clone_state_dict()

        self.model.load_state_dict(best_state_dict)
        self.model.eval().to(args.device, non_blocking=True)

        val_metrics = {"best-epoch": best_epoch, **self.evaluate(self.val_dataloader)}
        test_metrics = self.evaluate(self.test_dataloader)

        return val_metrics, test_metrics

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None)
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss, gold_labels, pred_labels = 0, [], []

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            out: SequenceClassifierOutput = self.model(**batch.to(self.args.device))

            batch_size: int = batch.input_ids.size(0)
            loss = out.loss.item() * batch_size
            total_loss += loss

            pred_labels += out.logits.argmax(dim=-1).tolist()
            gold_labels += batch.labels.tolist()

        accuracy: float = accuracy_score(gold_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=args.labels,
        )

        return {
            "loss": loss / len(dataloader.dataset),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"loss: {metrics['loss']:2.6f}   \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"precision: {metrics['precision']:.4f} \t"
            f"recall: {metrics['recall']:.4f} \t"
            f"f1: {metrics['f1']:.4f}"
        )

    def clone_state_dict(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}


def main(args: Args):
    exp = Experiment(args=args)
    val_metrics, test_metrics = exp.run()

    utils.save_json(val_metrics, args.output_dir / "val-metrics.json")
    utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args().parse_args()
    utils.init(seed=args.seed)
    main(args)
