# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : speed_augment.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/18 下午7:55
import random
import pydub
import librosa
import numpy as np
from tt.utils import read_wave_from_file, save_wav, tensor_to_img, get_feature


def speed_baidu(samples, min_speed=0.9, max_speed=1.1):
    """
    numpy线形插值速度增益
    :param samples: 音频数据，一维
    :param max_speed: 不能低于0.9，太低效果不好
    :param min_speed: 不能高于1.1，太高效果不好
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    speed = random.uniform(min_speed, max_speed)
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)
    return samples


def speed_librosa(samples, min_speed=0.9, max_speed=1.1):
    """
    librosa时间拉伸
    :param samples: 音频数据，一维
    :param max_speed: 不要低于0.9，太低效果不好
    :param min_speed: 不要高于1.1，太高效果不好
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype

    speed = random.uniform(min_speed, max_speed)
    samples = samples.astype(np.float)
    samples = librosa.effects.time_stretch(samples, speed)
    samples = samples.astype(data_type)
    return samples


# TODO: 效果不好
def speed_pydub(samples, min_speed=0.9, max_speed=1.1):
    """
    pydub变速
    :param samples: 音频数据，一维
    :param max_speed: 不要低于0.9，太低效果不好
    :param min_speed: 不要高于1.1，太高效果不好
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    speed = random.uniform(min_speed, max_speed)

    segment = pydub.AudioSegment(samples)
    samples = pydub.audio_segment.effects.speedup(segment, playback_speed=1.2)

    samples = samples.astype(data_type)
    return samples


if __name__ == '__main__':
    file = '../audio/dev.wav'
    audio_data, frame_rate = read_wave_from_file(file)
    feature = get_feature(audio_data)
    x = feature.shape[0]
    print('feature   ：', feature.shape)
    tensor_to_img(feature)

    audio_data = speed_librosa(audio_data)

    out_file = '../audio/speed_augment.wav'
    save_wav(out_file, audio_data)
    feature = get_feature(audio_data)
    print('feature   ：', feature.shape)
    tensor_to_img(feature, x_range=x)
