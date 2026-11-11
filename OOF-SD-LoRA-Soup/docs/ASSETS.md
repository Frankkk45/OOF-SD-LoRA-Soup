# Model and data setup / モデルとデータの準備

[English README](../README.md) · [日本語 README](../README.ja.md)

## English

This repository is the source release, not the complete local experiment archive. It includes 50 training configurations, 12 fold files and aggregate experiment records. It does not include pretrained checkpoints, trained adapters, dataset text, training logs, soft-target files, per-text predictions or the personal thesis documents. No hosted adapter download URL is configured in this release.

### Browse the reported results

After installing `requirements.txt`, run:

```bash
python tools/check_source.py
python tools/app.py --no-models
```

Open http://127.0.0.1:7860. This mode displays the reported results without loading weights and does not perform inference. The JSON summaries are the reported records; they are not a new reproduction of the experiment.

### Run inference

1. Obtain the Qwen3.5-4B revision listed in `model_manifest.json` and place the files under `models/Qwen3.5-4B/`. Alternatively, pass its directory with `--model-dir`.
2. Copy the trained adapter for the chosen domain from the full experiment archive to `weights/<domain>/sd_uniform6_soup/`. It must contain `adapter_config.json` and `adapter_model.safetensors`. A base-model download alone does not supply this project's trained adapters.
3. Run `python tools/app.py --device auto`, or use `tools/predict.py` as shown in the README. You do not need training or test datasets to classify your own text.

To download the pinned backbone after installing the requirements:

```bash
python - <<'PY'
import json
from huggingface_hub import snapshot_download
with open('model_manifest.json', encoding='utf-8') as handle:
    model = json.load(handle)['models']['Qwen3.5-4B']
snapshot_download(
    repo_id=model['repo_id'],
    revision=model['revision'],
    local_dir='models/Qwen3.5-4B',
)
PY
```

This downloads several gigabytes. `download_complete: false` describes the source release; the scorer checks actual local files rather than this flag. The manifest's file list identifies the files used for training provenance.

For the other models, restore the matching folders under `weights/<domain>/baseline/`, `members/`, `teachers/` or `a_uniform3_soup/`. The Thesis I models additionally require the Chinese RoBERTa backbone under `models/chinese-roberta-wwm-ext/` and the `.pth` files under `weights/thesis1/`.

### Prepare datasets

| Dataset | Required local files | Preparation |
|:--|:--|:--|
| usual / virus | `data/<domain>/train.tsv`, `dev.tsv`, `test.tsv`, `split_sizes.json` | Restore the exact files from the experiment archive. Dataset information: [SMP2020-EWECT](https://smp2020ewect.github.io/). No automatic reconstruction from arbitrary public copies is supplied. |
| GoEmotions | Original `train.tsv`, `dev.tsv`, `test.tsv`, `emotions.txt`, `ekman_mapping.json` in `data/goemotions/source/` | Obtain the upstream files for revision `2adf640a14f11025ae5a9d0ec493b78530d276d3` from [google-research/goemotions](https://github.com/google-research/google-research/tree/2adf640a14f11025ae5a9d0ec493b78530d276d3/goemotions). Preserve upstream notices, then run `python tools/build_goemotions.py`. |
| WRIME | Original `wrime-ver2.tsv` in `data/wrime/source/` | Obtain the unchanged release at revision `ac92deaff7845f31fdd46ef5d893300887006d13` from [ids-cv/wrime](https://github.com/ids-cv/wrime/tree/ac92deaff7845f31fdd46ef5d893300887006d13). Preserve upstream notices, then run `python tools/build_wrime.py`. |

The preprocessing scripts check source/output hashes. Their output format is UTF-8 TSV with the header `label<TAB>text_a`, plus a split-size JSON file. Use `python tools/build_wrime.py --protocol edom18` only for the alternative test-label protocol.

The provided folds are tied to the exact training-file hashes and row order. A different public conversion is not an interchangeable input. Do not publish the locally processed WRIME text splits; consult the retained [notice](../data/wrime/NOTICE.txt) and [license](../data/wrime/source/LICENSE).

### Train from the beginning

With the backbone and all splits for the selected domain restored:

```bash
python tools/train_pipeline.py --domain usual plan
python tools/train_pipeline.py --domain usual run
```

The complete pipeline generates new targets under `runs/targets/` before student training; it does not need the archived targets when started from the beginning. It requires the provided folds and configurations and a suitable NVIDIA GPU. Resuming directly at `SD_s42`, running the six teacher-control configurations, or reproducing the archived targets requires the appropriate existing targets and teacher outputs.

New adapters are written under `runs/`, not the demo's `weights/` paths. For example, after training usual, the final soup is in `runs/usual_sd_uniform6_soup/` and can be used directly:

```bash
python tools/predict.py --domain usual --model runs/usual_sd_uniform6_soup "今天终于完成了。"
```

Copy a completed adapter into the corresponding `weights/` directory if you want the demo to discover it.

### Recorded experiments

The full archive also contains `targets/`, `logs/`, `reference/predictions/` and all adapter weights, including ablation and Thesis I models. These are omitted from this source upload. Restore them from the experiment archive before using `tools/check_package.py`, `tools/build_soup.py --check-all`, or comparing archived per-text predictions. Without `logs/`, the training plan cannot display historical run durations.

`MANIFEST.sha256` here covers this source release only. `tools/check_source.py` checks its files, syntax and README links; it does not verify model behavior. The full archive's own manifest is needed to verify the archived assets against their original distribution.

## 日本語

このリポジトリはソース配布版です。50 個の学習設定、12 個の fold ファイル、集計結果を含みます。基盤モデル、学習済みアダプター、データ本文、学習ログ、ソフトターゲット、個別予測、個人の論文資料は含みません。この配布版にはアダプターの公開ダウンロード URL は設定されていません。

### 結果ページだけを確認する

依存パッケージをインストールした後、`python tools/check_source.py` でソースを検査し、`python tools/app.py --no-models` を実行してください。http://127.0.0.1:7860 で結果ページを確認できます。このモードは保存済みの結果を表示するもので、推論やスコアの再計算は行いません。

### 推論に必要なもの

`model_manifest.json` に記載された Qwen3.5-4B を `models/Qwen3.5-4B/` に配置します。上の Python コマンドで指定リビジョンを取得できます。さらに、完全な実験アーカイブから対象ドメインのアダプターを `weights/<domain>/sd_uniform6_soup/` に配置してください。必要なファイルは `adapter_config.json` と `adapter_model.safetensors` です。基盤モデルだけでは本研究の学習済み分類器にはなりません。

準備後は `python tools/app.py --device auto`、または README の予測コマンドを使えます。自分で入力した文章の分類には、学習・評価データセットは不要です。`download_complete: false` はソース配布版の状態を示し、推論コードはフラグではなく実際のファイルを確認します。

### データと再学習

- usual / virus：実験で使った `train.tsv`、`dev.tsv`、`test.tsv`、`split_sizes.json` を `data/<domain>/` に復元します。別の配布版は行順やハッシュが異なる場合があります。
- GoEmotions：上記の固定リビジョンから元ファイルを `data/goemotions/source/` に配置し、`python tools/build_goemotions.py` を実行します。
- WRIME：上記の固定リビジョンから変更していない `wrime-ver2.tsv` を `data/wrime/source/` に配置し、`python tools/build_wrime.py` を実行します。加工済みテキストは再配布しないでください。

fold は元の行順とハッシュに依存します。CUDA 環境、基盤モデル、対象データの全 split が揃えば、`tools/train_pipeline.py --domain usual run` で最初から学習できます。この場合、必要なソフトターゲットは `runs/targets/` に生成されます。途中再開や教師アブレーションには、それぞれ対応するターゲットと教師出力が別途必要です。

新しい学習結果は `runs/` に保存されます。デモで使う場合は対応する `weights/` に配置してください。CLI なら `--model runs/usual_sd_uniform6_soup` のように直接指定できます。

### 完全な実験検証

`tools/check_package.py` や `tools/build_soup.py --check-all` は、元の実験アーカイブのデータ、重み、ターゲット、個別予測などを復元してから実行してください。`logs/` がなければ、計画表示に過去の学習時間は表示されません。

この配布版の `MANIFEST.sha256` と `tools/check_source.py` はソースの整合性を確認するためのものです。学習済みモデルの動作や元アセットのハッシュを保証するものではありません。元アセットの検証には、完全なアーカイブ側の manifest を使ってください。
