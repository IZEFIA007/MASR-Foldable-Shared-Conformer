import argparse
import functools

from masr.trainerfusion import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformerfusion_split_ffn_kl01.yml', '配置文件')
add_arg("use_gpu",          bool,  True,                       '是否使用GPU评估模型')
add_arg('save_model',       str,   'models/foldable_split_ffn_kl01', '模型保存的路径')
add_arg('resume_model',     str,   'models/foldable_split_ffn_kl01/FoldableSplitSharedConformerModel_fbank/best_model/', '准备转换的模型路径')
args = parser.parse_args()
print_arguments(args=args)


# 获取训练器
trainer = MASRTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 导出预测模型
trainer.export(save_model_path=args.save_model,
               resume_model=args.resume_model)
