# BERT Classification Tutorial


## はじめに


## Installation & データセット準備

本実装は**Python 3.10以上**での実行を想定しています。
Python 3.10は、match文の導入やwith文の改善など様々な利便性の向上がなされている他、[Pythonが高速化の計画を進めていること](https://forest.watch.impress.co.jp/docs/news/1451751.html)もあり、早めに新しいPythonに適応しておくことのメリットは大きいと考えたためです。

また、Python 3.10では、type hints (型注釈)が以前のバージョンより自然に書けるようになっており、今までよりも堅牢かつ可読性の高いコードを書きやすくなっています。
そのため、公開実装のためのPythonとしても優れていると考えました。

### Install with poetry

```bash
poetry install
```

### Install with conda & pip


https://pytorch.org/get-started/locally/

```bash
conda create -n bert-classification-tutorial python=3.10

conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia

pip install tqdm "transformers[ja,sentencepiece]" classopt tokenizers numpy pandas more-itertools scikit-learn scipy
```


### データセット作成

本実装では、分類対象のテキストとしてRONDHUIT社が公開する[livedoorニュースコーパス](http://www.rondhuit.com/download.html#ldcc)を用います。
livedoorニュースコーパスは、9つのカテゴリのニュース記事が集約されたデータセットです。
通常、ニュース記事のタイトルと本文を用いて、そのニュース記事がどのカテゴリにあてはまるかを分類する9値分類を行います。

本実装では、以下のコマンドを実行すればデータセットの準備が完了するようになっています。


```bash
bash src/download.sh

poetry run python src/prepare.py
// python src/prepare.py
```


流れとしては、まず`src/download.sh`がデータセットのダウンロードと生データの展開を行います。

次に、`src/prepare.py`を実行することで、生データをJSONL形式(1行ごとにJSON形式のデータが書き込まれている形式)に変換します。
その際、NFKC正規化などの前処理も実行します。

さらに、分類モデルの訓練のため、分類先となるカテゴリを文字列から数値に変換し、その変換表を保存します。

また、全データを訓練(train):開発(val):テスト(test)=8:1:1の割合に分割します。
これにより、訓練中に開発セットを用いて、モデルが過学習していないかの確認が行えるようになります。
テストセットは最終的な評価にのみ用います。


### 訓練

以下のコマンドを実行することで、`cl-tohoku/bert-base-japanese-v2`を用いたテキスト分類モデルの訓練が実行できます。

```bash
poetry run python src/train.py --model_name cl-tohoku/bert-base-japanese-v2
```

この時、`--model_name`に与える引数を例えば`bert-base-multilingual-cased`にすることで、多言語BERTを用いた学習が実行できます。

また、ほとんどの設定をコマンドライン引数として与えら得れるようにしているので、以下のように複数の設定を変更して実行することも可能です。

```bash
poetry run python src/train.py \
  --model_name studio-ousia/luke-japanese-base-lite \
  --batch_size 32 \
  --epochs 10 \
  --lr 5e-5 \
  --num_warmup_epochs 1
```

本実装では学習後のモデルは保存せず、訓練のたびに評価値を算出し、評価値のみを保存するようにしています。
学習済みモデルを保存→保存済みモデルを読み込んで評価、という流れの実装をよく見ますが、この実装は実験の途中でどのモデルを使用していたのか忘れてしまったり、モデルの構造が学習時と変わってしまっていたり、評価用データを間違えてしまったり、といった問題が発生しやすいと考えています。

そこで本実装では、訓練のたびに必要な評価を行ってその結果のみを保存しておき、モデルは保存しない方針を採用しました。
これにより、モデルの構造を変化させたり、学習・評価データを変化させた場合でも、訓練をし直すだけで常に間違いのない結果を得られます。
研究における実験プロセスの中では、間違いのない実験結果を積み重ねていくことが、研究を進めていく上で最も重要だと考えているので、間違いが発生しづらいこの方針はスジがよいと考えています。

本実装において、実験結果は `outputs/[モデル名]/[年月日]/[時分秒]`のディレクトリに保存されます。
実際には、以下のようなディレクトリ構造になります。

```
outputs/bert-base-multilingual-cased
└── 2023-01-13
    └── 05-38-02
        ├── config.json
        ├── log.csv
        ├── test-metrics.json
        └── val-metrics.json
```

`config.json`が実験時の設定で、このファイルに記述してある値を用いることで、同じ実験を再現することができるようにしてあります。
また、`log.csv`に学習過程における開発セットでのepochごとの評価値を記録してあります。
そして、`val-metrics.json`と`test-metrics.json`に、開発セットの評価値が最もよかった時点でのモデルを用いた、開発セットとテストセットに対する評価値を記録してあります。

実際の`test-metrics.json`は以下のようになっています。

```json:test-metrics.json
{
  "loss": 2.845567681340744,
  "accuracy": 0.9619565217391305,
  "precision": 0.9561782755165722,
  "recall": 0.9562792450871114,
  "f1": 0.9559338777925345
}
```

## 評価実験

最後に、本実装によって、livedoorニュースコーパスの9値分類を行う評価実験を実施しました。

注意点ですが、実験は単一の乱数シード値で1度しか実施しておらず、分割交差検証も行っていないので、実験結果の正確性は高くありません。
したがって、以下の結果は過度に信用せず、参考程度に見てもらうよう、お願いいたします。

では、結果の表を以下に示します。
baseサイズのモデルとlargeサイズのモデルの2種類にモデルを大別して結果をまとめました。
なお、Accuracy (正解率)以外の値、つまりPresicion (精度)、Recall (再現率)、F1はmacro平均を取った値です。
また、すべての値は%表記です。

| base models                                                                                                               | Accuracy  | Precision | Recall    | F1        |
| ------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | --------- | --------- |
| [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)                                 | **97.15** | **96.82** | **96.55** | **96.64** |
| [cl-tohoku/bert-base-japanese-char-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v2)                       | 96.20     | 95.54     | 95.21     | 95.34     |
| [cl-tohoku/bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese)                                       | 96.47     | 96.15     | 95.67     | 95.83     |
| [cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) | 96.74     | 96.43     | 95.97     | 96.13     |
| [cl-tohoku/bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char)                             | 95.65     | 94.98     | 94.88     | 94.89     |
|                                                                                                                           |           |           |           |           |
| [studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)                       | 96.88     | 96.53     | 96.47     | 96.48     |
|                                                                                                                           |           |           |           |           |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                                       | 96.20     | 95.62     | 95.63     | 95.59     |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)                                                               | 96.20     | 95.65     | 95.60     | 95.61     |
| [studio-ousia/mluke-base-lite](https://huggingface.co/studio-ousia/mluke-base-lite)                                       | 96.47     | 95.82     | 95.94     | 95.86     |

まず、baseサイズのモデルの結果について観察すると、今回の実験では東北大BERTのバージョン2 (bert-base-japanese-v2)が最も高い性能になったことがわかります。
Accuracyが97.15、F1が96.64と、かなり高い割合で正しく分類することができていると思います。
東北大が公開しているモデルのbert-base-japanese-whole-word-maskingと比較して、bert-base-japanese-v2の方が性能が高く、東北大BERTの中だと、今後は最初にbert-base-japanese-v2を使って問題なさそうだという印象です。

次点はStudio Ousiaの日本語LUKEで、こちらも非常に高い割合で正しく分類を行えていると思います。

文字ベースのモデル(cl-tohoku/bert-base-japanese-char-v2など)は、他のモデルと比較して若干性能が低いですが、十分高い性能であるといえると思います。

多言語モデルの中では、Studio OusiaのmLUKEが最も高い性能になりました。


| large models                                                                                          | Accuracy  | Precision | Recall    | F1        |
| ----------------------------------------------------------------------------------------------------- | --------- | --------- | --------- | --------- |
| [cl-tohoku/bert-large-japanese](https://huggingface.co/cl-tohoku/bert-large-japanese)                 | **97.69** | **97.50** | 96.84     | **97.10** |
| [studio-ousia/luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite) | 97.55     | 97.38     | 96.85     | 97.06     |
|                                                                                                       |           |           |           |           |
| [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)                                         | 97.15     | 96.73     | 96.71     | 96.70     |
| [studio-ousia/mluke-large-lite](https://huggingface.co/studio-ousia/mluke-large-lite)                 | 97.42     | 97.25     | **96.97** | 97.08     |

次に、largeサイズのモデルの結果について観察すると、AccuracyとF1では東北大BERT (large)が最も高い性能になりましたが、Studio Ousiaの日本語LUKEや多言語LUKEと比較して、ほとんど同じ性能になりました。
全体として、baseサイズのモデルよりも高い性能となっており、モデルサイズを増大させることによる性能向上が観察できました。

## 参考文献

- [【実装解説】日本語版BERTでlivedoorニュース分類：Google Colaboratoryで（PyTorch）](https://qiita.com/sugulu_Ogawa_ISID/items/697bd03499c1de9cf082)
- [Livedoorニュースコーパスを文書分類にすぐ使えるように整形する](https://radiology-nlp.hatenablog.com/entry/2019/11/25/124219)
- https://github.com/yoheikikuta/bert-japanese/blob/master/notebook/finetune-to-livedoor-corpus.ipynb
- https://github.com/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb

## 引用

作者: [Hayato Tsukagoshi](https://hpprc.dev) \
email: [research.tsukagoshi.hayato@gmail.com](mailto:research.tsukagoshi.hayato@gmail.com)


```bibtex
@misc{
  hayato-tsukagoshi-2023-bert-classification-tutorial,
  title = {{BERT Classification Tutorial}},
  author = {Hayato Tsukagoshi},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hppRC/bert-classification-tutorial}},
  url = {https://github.com/hppRC/bert-classification-tutorial},
}
```