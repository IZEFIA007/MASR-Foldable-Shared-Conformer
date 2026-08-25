# MASR Foldable Shared Conformer

This repository is a Conformer-focused extension of
[yeyupiaoling/MASR](https://github.com/yeyupiaoling/MASR). It keeps the local
training, evaluation, export, and inference workflow of MASR while adding the
foldable and parameter-sharing models used in this project.

## Differences from MASR

- A foldable encoder maps 6 physical layers to as many as 12 logical layers.
- Maximum-depth and seed-depth paths can be trained jointly with self-distillation.
- Adjacent physical layers can share grouped parameters.
- The Macaron pre-FFN and post-convolution FFN can use separate parameter sets.
- Relative positional encoding is retained in the final shared models.
- The KL term can be controlled without adding the raw KL loss twice in the trainer.

## Included models

| Experiment | Model name in YAML | Config | Trainer |
| --- | --- | --- | --- |
| Conformer baseline | `ConformerModel` | `configs/conformer1.yml` | `masr.trainer` |
| Foldable Conformer | `FoldableConformerModel` | `configs/conformerdis.yml` | `masr.trainerdis` |
| Group-shared Conformer | `SharedConformerModel` | `configs/conformershare.yml` | `masr.trainer` |
| Foldable + shared, rel-pos | `FoldableSharedConformerModel` | `configs/conformerfusion_relpos.yml` | `masr.trainerfusion` |
| Split Macaron FFNs, historical KL | `FoldableSplitSharedConformerModel` | `configs/conformerfusion_split_ffn.yml` | `masr.trainerfusion` |
| Split Macaron FFNs, effective KL=0.1 | `FoldableSplitSharedConformerModel` | `configs/conformerfusion_split_ffn_kl01.yml` | `masr.trainerfusion` |

## Installation

```bash
conda create -n masr python=3.11
conda activate masr
pip install -r requirements.txt
pip install -e .
```

Prepare the dataset manifests, vocabulary, and CMVN statistics with
`create_data.py`, then update the dataset paths in the selected YAML file.

## Switching models

Model selection has three parts:

1. The model class must be registered in `masr/model_utils/__init__.py`.
2. `model_conf.model` in the YAML selects the registered model.
3. `train.py` and `eval.py` must import the trainer listed in the table above.

For example, use this import for the foldable model:

```python
from masr.trainerdis import MASRTrainer
```

Use this import for the shared foldable and Split-FFN models:

```python
from masr.trainerfusion import MASRTrainer
```

## Training

```bash
python train.py \
  --configs=configs/conformerfusion_split_ffn_kl01.yml \
  --save_model_path=models/foldable_split_ffn_kl01 \
  --log_dir=log/foldable_split_ffn_kl01
```

For another experiment, switch the trainer import and pass the corresponding
configuration from the table.

## Evaluation

```bash
python eval.py \
  --configs=configs/conformerfusion_split_ffn_kl01.yml \
  --resume_model=models/foldable_split_ffn_kl01/FoldableSplitSharedConformerModel_fbank/best_model \
  --decoder=ctc_greedy_search
```

`evaluate_matrix.py` evaluates the included model families with one CTC-greedy
code path. Its checkpoint paths can be adjusted in the `MODELS` table.

## Export and local inference

```bash
python export_model.py --help
python infer_path.py --help
```

Export a trained checkpoint first, then pass the resulting inference-model
directory and an audio file to `infer_path.py`.

## License

The upstream MASR code and this derivative repository are distributed under
the Apache License 2.0. See `LICENSE` and `NOTICE.md`.
