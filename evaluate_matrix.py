import argparse
import gc
import json
import os

import sentencepiece

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from masr.data_utils.audio_featurizer import AudioFeaturizer
from masr.data_utils.collate_fn import collate_fn
from masr.data_utils.reader import MASRDataset
from masr.data_utils.tokenizer import MASRTokenizer
from masr.decoders.ctc_greedy_search import ctc_greedy_search
from masr.model_utils.conformer.model import ConformerModel as StandardConformerModel
from masr.model_utils.conformershare.model import ConformerModel as SharedConformerModel
from masr.model_utils.conformerdis.model import FoldableConformerModel as FoldableModel
from masr.model_utils.conformerfusion.model import FoldableConformerModel as FoldableSharedModel
from masr.utils.metrics import cer
from masr.utils.utils import dict_to_object


MODELS = [
    {
        "name": "baseline_conformer",
        "config": "configs/conformer1.yml",
        "checkpoint": "modelsai1/ConformerModel_fbank/best_model",
        "depths": [None],
    },
    {
        "name": "foldable_conformer",
        "config": "configs/conformerdis.yml",
        "checkpoint": "modelsaidis1/FoldableConformerModel_fbank/best_model",
        "depths": [6, 8, 10, 12],
    },
    {
        "name": "foldable_shared_relpos_final",
        "config": "configs/conformerfusion_relpos.yml",
        "checkpoint": "modelsfusion_relpos/FoldableSharedConformerModel_fbank/best_model",
        "depths": [12],
    },
    {
        "name": "foldable_split_ffn",
        "config": "configs/conformerfusion_split_ffn.yml",
        "checkpoint": "modelsfusion_split_ffn/FoldableSplitSharedConformerModel_fbank/best_model",
        "depths": [12],
    },
    {
        "name": "foldable_split_ffn_kl01",
        "config": "configs/conformerfusion_split_ffn_kl01.yml",
        "checkpoint": "modelsfusion_split_ffn_kl01/FoldableSplitSharedConformerModel_fbank/best_model",
        "depths": [12],
    },
    {
        "name": "shared_conformer",
        "config": "configs/conformershare.yml",
        "checkpoint": "modelsaishare/ConformerModel_fbank/best_model",
        "depths": [None],
    },
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return dict_to_object(data)


def choose_model_class(state_dict):
    has_shared = any("encoder.shared_ffn_" in k for k in state_dict)
    has_physical = any(k.startswith("encoder.physical_layers.") for k in state_dict)
    if has_physical and has_shared:
        return FoldableSharedModel
    if has_physical:
        return FoldableModel
    if has_shared:
        return SharedConformerModel
    return StandardConformerModel


def build_model(config, tokenizer, checkpoint_path, device="cpu"):
    state_dict = torch.load(os.path.join(checkpoint_path, "model.pth"), map_location="cpu", weights_only=True)
    model_cls = choose_model_class(state_dict)

    encoder_conf = config.encoder_conf
    if model_cls in (StandardConformerModel, SharedConformerModel):
        num_blocks = getattr(getattr(encoder_conf, "encoder_args", None), "num_blocks", None)
        group_count = 0
        for k in state_dict:
            if k.startswith("encoder.shared_ffn_w1."):
                try:
                    group_count = max(group_count, int(k.rsplit(".", 1)[1]) + 1)
                except ValueError:
                    pass
        if group_count and num_blocks:
            encoder_conf.encoder_args.group_size = num_blocks // group_count

    model_args = dict(config.model_conf.model_args)
    model_args.update({
        "input_size": 80,
        "vocab_size": tokenizer.vocab_size,
        "mean_istd_path": config.dataset_conf.mean_istd_path,
        "eos_id": tokenizer.eos_id,
    })
    model = model_cls(encoder_conf=encoder_conf, decoder_conf=config.decoder_conf, **model_args)
    model = model.to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch for {checkpoint_path}: missing={missing_keys}, unexpected={unexpected_keys}")
    return model


def build_loader(config, tokenizer, audio_featurizer, manifest_path, limit=None, batch_size=None):
    dataset_args = dict(config.dataset_conf.dataset)
    dataset = MASRDataset(
        data_manifest=manifest_path,
        audio_featurizer=audio_featurizer,
        tokenizer=tokenizer,
        mode="eval",
        **dataset_args,
    )
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size or config.dataset_conf.batch_sampler.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )
    return loader


def evaluate_model(model, tokenizer, loader, depths, device="cpu"):
    model.eval()
    error_results = {depth: [] for depth in depths}
    with torch.no_grad():
        for inputs, labels, input_lens, label_lens in tqdm(loader, desc=f"depths={depths}", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_lens = input_lens.to(device)
            label_ids = labels.cpu().numpy().tolist()
            label_texts = tokenizer.ids2text([list(filter(lambda x: x != -1, ids)) for ids in label_ids])
            for depth in depths:
                if depth is None:
                    _, ctc_probs, ctc_lens = model.get_encoder_out(inputs, input_lens)
                else:
                    _, ctc_probs, ctc_lens = model.get_encoder_out(inputs, input_lens, logical_depth=depth)
                pred_ids = ctc_greedy_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens, blank_id=tokenizer.blank_id)
                pred_texts = tokenizer.ids2text([r for r in pred_ids])
                for label, pred in zip(label_texts, pred_texts):
                    error_results[depth].append(cer(label, pred))
    return {depth: (sum(v) / len(v) if v else -1.0) for depth, v in error_results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_manifest", default="dataset1/eval.jsonl")
    parser.add_argument("--test_manifest", default="dataset1/test.jsonl")
    parser.add_argument("--out", default="evaluation_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", default=None, help="comma-separated model names")
    parser.add_argument("--splits", default=None, help="comma-separated eval/test")
    parser.add_argument("--depths", default=None, help="comma-separated depths or all")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_file = open(args.out, "w", encoding="utf-8")
    for spec in MODELS:
        if args.models and spec["name"] not in args.models.split(","):
            continue
        config = load_config(spec["config"])
        config.dataset_conf.mean_istd_path = "dataset1/mean_istd.json"
        config.tokenizer_conf.vocab_model_dir = "dataset1/vocab_model/"
        tokenizer = MASRTokenizer(**config.tokenizer_conf)
        audio_featurizer = AudioFeaturizer(
            feature_method=config.preprocess_conf.feature_method,
            method_args=config.preprocess_conf.get("method_args", {}),
        )
        model = build_model(config, tokenizer, spec["checkpoint"], device=args.device)
        params = sum(p.numel() for p in model.parameters())
        file_size = os.path.getsize(os.path.join(spec["checkpoint"], "model.pth"))

        splits = [("eval", args.eval_manifest), ("test", args.test_manifest)]
        if args.splits:
            allowed = set(args.splits.split(","))
            splits = [s for s in splits if s[0] in allowed]
        for split, manifest in splits:
            config.dataset_conf.eval_manifest = manifest
            loader = build_loader(config, tokenizer, audio_featurizer, manifest, limit=args.limit, batch_size=args.batch_size)
            depths = spec["depths"]
            if args.depths and args.depths != "all":
                allowed_depths = set(int(x) if x != "None" else None for x in args.depths.split(","))
                depths = [d for d in depths if d in allowed_depths]
            values = evaluate_model(model, tokenizer, loader, depths=depths, device=args.device)
            for depth, value in values.items():
                result = {
                    "model": spec["name"],
                    "split": split,
                    "depth": depth,
                    "cer": round(value, 6),
                    "params": params,
                    "model_pth_bytes": file_size,
                }
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
        del model
        try:
            del loader
        except Exception:
            pass
        gc.collect()

    out_file.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
