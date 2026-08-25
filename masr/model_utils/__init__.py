from loguru import logger

from masr.model_utils.conformer.model import ConformerModel
from masr.model_utils.conformershare.model import ConformerModel as SharedConformerModel
from masr.model_utils.conformerdis.model import FoldableConformerModel
from masr.model_utils.conformerfusion.model import FoldableConformerModel as FoldableSharedConformerModel
from masr.model_utils.conformerfusion.model import FoldableConformerModel as FoldableSplitSharedConformerModel

__all__ = ['build_model']


MODEL_CLASSES = {
    'ConformerModel': ConformerModel,
    'SharedConformerModel': SharedConformerModel,
    'FoldableConformerModel': FoldableConformerModel,
    'FoldableSharedConformerModel': FoldableSharedConformerModel,
    'FoldableSplitSharedConformerModel': FoldableSplitSharedConformerModel,
}


def build_model(input_size, vocab_size, mean_istd_path, eos_id, encoder_conf, decoder_conf, model_conf):
    use_model = model_conf.get('model', 'ConformerModel')
    model_args = model_conf.get('model_args', {})
    model_args.input_size = input_size
    model_args.vocab_size = vocab_size
    model_args.mean_istd_path = mean_istd_path
    model_args.vocab_size = vocab_size
    model_args.eos_id = eos_id
    model_cls = MODEL_CLASSES.get(use_model)
    if model_cls is None:
        raise ValueError(f"未知模型名称：{use_model}")
    model = model_cls(encoder_conf=encoder_conf, decoder_conf=decoder_conf, **model_args)
    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
