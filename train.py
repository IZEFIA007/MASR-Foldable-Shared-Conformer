import argparse
import functools
import os
import torch

from masr.trainerfusion import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('configs',              str,    'configs/conformerfusion_split_ffn_kl01.yml', '配置文件')
add_arg('data_augment_configs', str,    'configs/augmentation.yml',     '数据增强配置文件')
add_arg("local_rank",           int,    0,                              '多卡训练的本地GPU')
add_arg("use_gpu",              bool,   True,                           '是否使用GPU训练')
add_arg('metrics_type',         str,    'cer',                          '评估指标类型')
add_arg('save_model_path',      str,    'models/foldable_split_ffn_kl01', '模型保存路径')
add_arg('log_dir',              str,    'log/foldable_split_ffn_kl01',    '日志路径')
add_arg('resume_model',         str,    None,                           '恢复训练模型路径')
add_arg('pretrained_model',     str,    None,                           '预训练模型路径')
add_arg('overwrites',           str,    None,                           '覆盖配置参数')

args = parser.parse_args()

print(f"args.resume_model = {args.resume_model}, type = {type(args.resume_model)}")
if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    print_arguments(args=args)

trainer = MASRTrainer(
    configs=args.configs,
    use_gpu=args.use_gpu,
    metrics_type=args.metrics_type,
    data_augment_configs=args.data_augment_configs,
    overwrites=args.overwrites
)

trainer.train(
    save_model_path=args.save_model_path,
    log_dir=args.log_dir,
    resume_model=args.resume_model,
    pretrained_model=args.pretrained_model
)
