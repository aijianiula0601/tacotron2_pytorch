import os
import sys
import numpy as np
from pathlib import Path
from tqdm import trange
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_dir)
print('prj_dir:', project_dir)

from utils.audio import melspectrogram, load_wav


def files_to_list(fdir):
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            f_list.append([wav_path, parts[1]])
    return f_list


def get_mel(filename):
    wav = load_wav(filename)
    mel = melspectrogram(wav).astype(np.float32)
    return mel


def get_mel_text_pair(filename_and_text):
    filename, text = filename_and_text[0], filename_and_text[1]
    mel = get_mel(filename)
    return text, mel


def process_groups(files_cl, save_mel_dir):
    w_lines_cl = []
    for line in files_cl:
        wav_path, text = line[0], line[1]
        mel = get_mel(wav_path)
        save_mel_path = str(Path(save_mel_dir).joinpath(Path(wav_path).name.replace(".wav", ".npy")))
        np.save(save_mel_path, mel)
        w_line = "{}|{}\n".format(Path(save_mel_path).name, text)
        w_lines_cl.append(w_line)
    return w_lines_cl


if __name__ == '__main__':
    """
    注意：ljspeech数据集的数据采用率为22050，提取的mel频谱也是采用22050
    """
    fdir = '/home/huangjiahong/tmp/common_dataset/LJSpeech-1.1'
    save_mel_dir = '/home/huangjiahong/tmp/tts/dataset/api/real_time_voice_clone_dataset/ljspeech/mels'
    meta_file = '/home/huangjiahong/tmp/tts/dataset/api/real_time_voice_clone_dataset/ljspeech/train.txt'
    filelines = files_to_list(fdir)

    group_num = len(filelines) // 1000
    lines_groups = [filelines[i:i + group_num] for i in range(0, len(filelines), group_num)]
    executor = ProcessPoolExecutor(max_workers=4)
    all_task = [executor.submit(partial(process_groups, files_cl, save_mel_dir)) for files_cl in lines_groups]

    with open(meta_file, 'w', encoding='utf-8') as f:
        for task in tqdm(all_task):
            lines = task.result()
            for line in lines:
                f.write(line)
