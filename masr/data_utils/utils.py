import json
import os
import time

import numpy as np
import soundfile
from loguru import logger
from tqdm import tqdm
from yeaudio.audio import AudioSegment
from zhconv import convert

from masr.data_utils.binary import DatasetWriter


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.0):
    """读取数据列表文件

    :param manifest_path: 数据列表的路径
    :type manifest_path: str
    :param max_duration: 过滤的最长音频长度
    :type max_duration: float
    :param min_duration: 过滤的最短音频长度
    :type min_duration: float
    :return: 数据列表，JSON格式
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in open(manifest_path, 'r', encoding='utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if max_duration >= json_data["duration"] >= min_duration:
            manifest.append(json_data)
    return manifest


def create_manifest(annotation_path,
                    train_manifest_path,
                    test_manifest_path,
                    eval_manifest_path,
                    max_test_manifest=10000):
    """Create train/dev/test manifests without changing the dataset split.

    :param annotation_path: 标注列表文件夹路径
    :type annotation_path: str
    :param train_manifest_path: 训练数据列表路径
    :type train_manifest_path: str
    :param test_manifest_path: 测试数据列表路径
    :type test_manifest_path: str
    :param eval_manifest_path: 验证数据列表路径
    :type eval_manifest_path: str
    :param max_test_manifest: Deprecated compatibility argument. Official splits are never resampled.
    :type max_test_manifest: int
    """
    del max_test_manifest
    train_list = []
    test_list = []
    eval_list = []
    durations = []

    split_aliases = {
        'train': 'train',
        'dev': 'eval',
        'eval': 'eval',
        'valid': 'eval',
        'validation': 'eval',
        'test': 'test',
    }

    def get_split(filename):
        stem = os.path.splitext(filename)[0].lower().replace('-', '_')
        for prefix, split in split_aliases.items():
            if stem == prefix or stem.startswith(f'{prefix}_'):
                return split
        return None

    def append_item(split, item):
        if split == 'train':
            train_list.append(item)
        elif split == 'eval':
            eval_list.append(item)
        else:
            test_list.append(item)

    for annotation_text in sorted(os.listdir(annotation_path)):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        if not os.path.isfile(annotation_text_path):
            continue
        extension = os.path.splitext(annotation_text_path)[-1].lower()
        if extension not in {'.json', '.jsonl', '.txt'}:
            continue
        split = get_split(annotation_text)
        if split is None:
            logger.warning(f'无法从文件名判断数据划分，已跳过：{annotation_text}')
            continue

        # ----------------------------
        # 处理 JSON 标注
        # ----------------------------
        if extension in {'.json', '.jsonl'}:
            with open(annotation_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                try:
                    d = json.loads(line)
                except Exception as e:
                    logger.warning(f'{line} 错误，已跳过，错误信息：{e}')
                    continue

                audio_path, text = d["audio_filepath"], d["text"]
                duration = d["duration"]
                durations.append(duration)
                text = text.lower().strip()
                if len(text) == 0: continue
                text = convert(text, 'zh-cn')

                item = dict(
                    audio_filepath=audio_path.replace('\\', '/'),
                    text=text,
                    duration=duration
                )
                if 'start_time' in d and 'end_time' in d:
                    item['start_time'] = d['start_time']
                    item['end_time'] = d['end_time']
                append_item(split, item)

        # ----------------------------
        # 处理 txt 标注
        # ----------------------------
        else:
            with open(annotation_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in tqdm(lines):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) not in {2, 3}:
                        raise ValueError('TXT标注必须是 audio_path<TAB>text[<TAB>duration]')
                    audio_path, text = parts[:2]
                    duration = float(parts[2]) if len(parts) == 3 else None
                except Exception as e:
                    logger.warning(f'{line} 错误，已跳过，错误信息：{e}')
                    continue

                if duration is None:
                    audio_segment = AudioSegment.from_file(audio_path)
                    duration = audio_segment.duration
                durations.append(duration)

                text = text.lower().strip()
                if len(text) == 0 or text == ' ': continue
                text = convert(text, 'zh-cn')

                item = dict(
                    audio_filepath=audio_path.replace('\\', '/'),
                    text=text,
                    duration=duration
                )
                append_item(split, item)

    # ----------------------------
    # 排序
    # ----------------------------
    split_lists = {'train': train_list, 'eval/dev': eval_list, 'test': test_list}
    missing = [name for name, items in split_lists.items() if not items]
    if missing:
        raise ValueError(f'缺少非空数据划分：{", ".join(missing)}。请提供 train、dev/eval/valid 和 test 标注文件。')
    for items in split_lists.values():
        items.sort(key=lambda x: x["duration"], reverse=False)

    # ----------------------------
    # 写入文件
    # ----------------------------
    outputs = {
        train_manifest_path: train_list,
        eval_manifest_path: eval_list,
        test_manifest_path: test_list,
    }
    for manifest_path, items in outputs.items():
        parent = os.path.dirname(manifest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
            for item in items:
                manifest_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f'完成生成数据列表：train={len(train_list)}, eval/dev={len(eval_list)}, '
                f'test={len(test_list)}，总长度={sum(durations) / 3600.:.2f}小时')



def merge_audio(annotation_path, save_audio_path, max_duration=600, target_sr=16000):
    """将多段短音频合并为长音频，减少文件数量

    :param annotation_path: 标注列表文件夹路径
    :type annotation_path: str
    :param save_audio_path: 合并后的音频保存路径
    :type save_audio_path: str
    :param max_duration: 合并的最大音频长度
    :type max_duration: int
    :param target_sr: 目标采样率
    :type target_sr: int
    """
    # 合并数据列表
    train_list_path = os.path.join(annotation_path, 'merge_audio.json')
    if os.path.exists(train_list_path):
        f_ann = open(train_list_path, 'a', encoding='utf-8')
    else:
        f_ann = open(train_list_path, 'w', encoding='utf-8')
    wav, duration_sum, list_data = [], [], []
    for annotation_text in os.listdir(annotation_path):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        if os.path.splitext(annotation_text_path)[-1] != '.txt': continue
        if os.path.splitext(annotation_text_path)[-1] == 'test.txt': continue
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path, text = line.replace('\n', '').replace('\r', '').split('\t')
            if not os.path.exists(audio_path): continue
            audio_segment = AudioSegment.from_file(audio_path)
            # 重采样
            if audio_segment.sample_rate != target_sr:
                audio_segment.resample(target_sample_rate=target_sr)
            # 合并数据
            duration_sum.append(audio_segment.duration)
            wav.append(audio_segment.samples)
            # 列表数据
            list_d = dict(text=text,
                          duration=round(audio_segment.duration, 5),
                          start_time=round(sum(duration_sum) - audio_segment.duration, 5),
                          end_time=round(sum(duration_sum), 5))
            list_data.append(list_d)
            # 删除已处理的音频文件
            os.remove(audio_path)
            # 保存合并音频文件
            if sum(duration_sum) >= max_duration:
                # 保存路径
                dir_num = len(os.listdir(save_audio_path)) - 1 if os.path.exists(save_audio_path) else 0
                save_dir = os.path.join(save_audio_path, str(dir_num))
                os.makedirs(save_dir, exist_ok=True)
                if len(os.listdir(save_dir)) >= 1000:
                    save_dir = os.path.join(save_audio_path, str(dir_num + 1))
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.wav').replace('\\', '/')
                data = np.concatenate(wav)
                soundfile.write(save_path, data=data, samplerate=target_sr, format='WAV')
                # 写入到列表文件
                for list_d in list_data:
                    list_d['audio_filepath'] = save_path
                    f_ann.write('{}\n'.format(json.dumps(list_d)))
                f_ann.flush()
                wav, duration_sum, list_data = [], [], []
        # 删除已处理的标注文件
        os.remove(annotation_text_path)
    f_ann.close()


def create_manifest_binary(train_manifest_path, eval_manifest_path, test_manifest_path):
    """生成数据列表的二进制文件

    :param train_manifest_path: 训练列表的路径
    :type train_manifest_path: str
    :param eval_manifest_path: 验证列表的路径
    :param test_manifest_path: 测试列表的路径
    :type test_manifest_path: str
    """
    for manifest_path in [train_manifest_path, eval_manifest_path, test_manifest_path]:
        dataset_writer = DatasetWriter(manifest_path)
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.replace('\n', '')
            dataset_writer.add_data(line)
        dataset_writer.close()
