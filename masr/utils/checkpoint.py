import json
import os
import shutil

import torch
from loguru import logger
from masr import __version__


def load_pretrained(model, pretrained_model):
    if pretrained_model is None:
        return model

    if os.path.isdir(pretrained_model):
        pretrained_model = os.path.join(pretrained_model, 'model.pth')

    assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_state_dict = torch.load(pretrained_model, map_location=device)

    # 过滤 shape 不匹配参数
    for name, weight in list(model_dict.items()):
        if name in model_state_dict and list(weight.shape) != list(model_state_dict[name].shape):
            logger.warning(
                f'{name} shape mismatch: {list(model_state_dict[name].shape)} vs {list(weight.shape)}'
            )
            model_state_dict.pop(name, None)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        missing_keys, unexpected_keys = model.module.load_state_dict(model_state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    logger.info(f"Pretrain loaded. missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
    return model


def load_checkpoint(configs, model, optimizer, amp_scaler, scheduler,
                    step_epoch, save_model_path, resume_model):

    last_epoch1 = 0
    error_rate1 = 1.0

    def load_model(model_path):
        model_file = os.path.join(model_path, 'model.pth')
        opt_file = os.path.join(model_path, 'optimizer.pth')
        state_file = os.path.join(model_path, 'model.state')
        scaler_file = os.path.join(model_path, 'scaler.pth')
        scheduler_file = os.path.join(model_path, 'scheduler.pth')

        assert os.path.exists(model_file), f"缺少 model.pth: {model_path}"
        assert os.path.exists(opt_file), f"缺少 optimizer.pth: {model_path}"
        assert os.path.exists(state_file), f"缺少 model.state: {model_path}"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== 1. load model =====
        state_dict = torch.load(model_file, map_location=device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        logger.info(f"[CKPT] model restored")

        # ===== 2. load optimizer =====
        optimizer.load_state_dict(torch.load(opt_file, map_location=device))
        logger.info(f"[CKPT] optimizer restored")

        # 修复 optimizer state device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        # ===== 3. load AMP scaler =====
        if amp_scaler is not None and os.path.exists(scaler_file):
            amp_scaler.load_state_dict(torch.load(scaler_file, map_location=device))
            logger.info(f"[CKPT] scaler restored")

        # ===== 4. load meta =====
        with open(state_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        last_epoch = meta.get('last_epoch', 0)
        error_rate = meta.get('cer', meta.get('wer', meta.get('mer', 1.0)))
        logger.info(f"[CKPT] restored last_epoch={last_epoch}, error={error_rate}")

        # ===== 5. scheduler 对齐 =====
        if scheduler is not None:
            if os.path.exists(scheduler_file):
                scheduler.load_state_dict(torch.load(scheduler_file, map_location=device))
                logger.info("[CKPT] scheduler state restored")
            else:
                if hasattr(scheduler, "last_epoch"):
                    scheduler.last_epoch = last_epoch * step_epoch
                    logger.warning("[CKPT] scheduler missing, fallback to last_epoch alignment")
                else:
                    for _ in range(last_epoch * step_epoch):
                        scheduler.step()

        return last_epoch, error_rate

    save_feature_method = configs.preprocess_conf.feature_method
    last_model_dir = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'last_model')

    if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth')) and
                                    os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
        if resume_model is not None:
            last_epoch1, error_rate1 = load_model(resume_model)
        else:
            try:
                last_epoch1, error_rate1 = load_model(last_model_dir)
            except Exception as e:
                logger.warning(f"auto resume failed: {e}")

    return model, optimizer, amp_scaler, scheduler, last_epoch1, error_rate1


def save_checkpoint(configs, model, optimizer, amp_scaler,
                    save_model_path, epoch_id,
                    error_rate=1.0, metrics_type=None,
                    best_model=False, scheduler=None):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    save_feature_method = configs.preprocess_conf.feature_method

    if best_model:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'best_model')
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  f'epoch_{epoch_id}')
    os.makedirs(model_path, exist_ok=True)

    torch.save(state_dict, os.path.join(model_path, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
    if amp_scaler is not None:
        torch.save(amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(model_path, 'scheduler.pth'))

    data = {"last_epoch": epoch_id, "version": __version__, "model": configs.model_conf.model,
            "feature_method": save_feature_method}
    if metrics_type is not None:
        data[metrics_type] = error_rate

    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # ===== last_model 自动更新 =====
    if not best_model:
        last_model_path = os.path.join(save_model_path,
                                       f'{configs.model_conf.model}_{save_feature_method}',
                                       'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)

    logger.info(f"saved model: {model_path}")