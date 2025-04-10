# import os

# import numpy as np
# import pandas
# from augment.audio_augment import audio_augment
# import tt.kaldi_io as kaldi_io
# from tt.utils import get_feature, get_feature2, read_wave_from_file, concat_frame, subsampling


# class Dataset:
#     def __init__(self, config, type, dic):
#         self.type = type
#         self.word2index = dic
#         self.name = config.name
#         self.feature_dim = config.feature_dim
#         self.left_context_width = config.left_context_width
#         self.right_context_width = config.right_context_width
#         self.subsample = config.subsample
#         self.apply_cmvn = config.apply_cmvn
#         self.max_input_length = config.max_input_length
#         self.max_target_length = config.max_target_length
#         self.ignore_id = config.ignore_id

#         # self.arkscp = os.path.join(config.__getattr__(type), 'feats.scp')

#         if self.apply_cmvn:
#             self.utt2spk = {}
#             with open(os.path.join(config.__getattr__(type), 'utt2spk'), 'r') as fid:
#                 for line in fid:
#                     parts = line.strip().split()
#                     self.utt2spk[parts[0]] = parts[1]
#             self.cmvnscp = os.path.join(config.__getattr__(type), 'cmvn.scp')
#             self.cmvn_stats_dict = {}
#             self.get_cmvn_dict()

#         # self.feats_list, self.feats_dict = self.get_feats_list()

#     def __len__(self):
#         raise NotImplementedError

#     def pad(self, inputs, max_length=None):
#         dim = len(inputs.shape)
#         if dim == 1:
#             if max_length is None:
#                 max_length = self.max_target_length
#             pad_zeros_mat = np.zeros([1, max_length - inputs.shape[0]], dtype=np.int32)
#             # todo: replace 0 to  IGNOREID
#             pad_zeros_mat = pad_zeros_mat + self.ignore_id
#             padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
#         elif dim == 2:
#             if max_length is None:
#                 max_length = self.max_input_length
#             feature_dim = inputs.shape[1]
#             pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
#             padded_inputs = np.row_stack([inputs, pad_zeros_mat])
#         else:
#             raise AssertionError(
#                 'Features in inputs list must be one vector or two dimension matrix! ')
#         return padded_inputs

#     def get_cmvn_dict(self):
#         cmvn_reader = kaldi_io.read_mat_scp(self.cmvnscp)
#         for spkid, stats in cmvn_reader:
#             self.cmvn_stats_dict[spkid] = stats

#     def cmvn(self, mat, stats):
#         mean = stats[0, :-1] / stats[0, -1]
#         variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
#         return np.divide(np.subtract(mat, mean), np.sqrt(variance))


# class AudioDataset(Dataset):
#     def __init__(self, config, type, dic):
#         super(AudioDataset, self).__init__(config, type, dic)
#         self.config = config
#         self.short_first = config.short_first
#         self.df = pandas.read_csv(config.__getattr__(type), index_col=False)

#     def __getitem__(self, index):

#         audio_path = self.df.iloc[index, 0]
#         label = self.df.iloc[index, 1]

#         targets = np.array(self.encode(label))
#         wave_data, frame_rate = read_wave_from_file(audio_path)
#         # wave_data = audio_augment(wave_data)
#         features = get_feature(wave_data, frame_rate, self.feature_dim)
#         # features = get_feature2(wave_data, frame_rate, self.feature_dim)
#         # features = np.load(feats_scp)
#         features = concat_frame(features, self.left_context_width, self.right_context_width)
#         features = subsampling(features, self.subsample)

#         inputs_length = np.array(features.shape[0]).astype(np.int64)
#         targets_length = np.array(targets.shape[0]).astype(np.int64)

#         features = self.pad(features).astype(np.float32)
#         targets = self.pad(targets).astype(np.int64).reshape(-1)

#         return features, inputs_length, targets, targets_length

#     def __len__(self):
#         return self.df.shape[0]

#     def encode(self, seq):
#         # encoded_seq = [self.word2index.get('<SOS>')]
#         encoded_seq = []
#         for unit in seq:
#             if unit in self.word2index:
#                 encoded_seq.append(self.word2index[unit])
#             else:
#                 encoded_seq.append(self.word2index['<unk>'])  
#         return encoded_seq

import os
import numpy as np
import pandas
from augment.audio_augment import audio_augment
import tt.kaldi_io as kaldi_io
from tt.utils import get_feature, get_feature2, read_wave_from_file, concat_frame, subsampling

class Dataset:
    def __init__(self, config, type, dic):
        self.type = type
        self.word2index = dic
        self.name = config.name
        self.feature_dim = config.feature_dim
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.subsample = config.subsample
        self.apply_cmvn = config.apply_cmvn
        self.ignore_id = config.ignore_id  # ID của token <pad>

        if self.apply_cmvn:
            self.utt2spk = {}
            with open(os.path.join(config.__getattr__(type), 'utt2spk'), 'r') as fid:
                for line in fid:
                    parts = line.strip().split()
                    self.utt2spk[parts[0]] = parts[1]
            self.cmvnscp = os.path.join(config.__getattr__(type), 'cmvn.scp')
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

    def __len__(self):
        raise NotImplementedError

    def get_cmvn_dict(self):
        cmvn_reader = kaldi_io.read_mat_scp(self.cmvnscp)
        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))

class AudioDataset(Dataset):
    def __init__(self, config, type, dic):
        super(AudioDataset, self).__init__(config, type, dic)
        self.df = pandas.read_csv(config.__getattr__(type), index_col=False)

    def __getitem__(self, index):
        audio_path = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]

        # Trích xuất đặc trưng và label (không padding)
        targets = np.array(self.encode(label))
        wave_data, frame_rate = read_wave_from_file(audio_path)
        features = get_feature(wave_data, frame_rate, self.feature_dim)
        features = concat_frame(features, self.left_context_width, self.right_context_width)
        features = subsampling(features, self.subsample)

        # Trả về dữ liệu gốc (không pad)
        return {
            "features": features.astype(np.float32),
            "features_len": features.shape[0],
            "targets": targets.astype(np.int64),
            "targets_len": targets.shape[0]
        }

    def __len__(self):
        return self.df.shape[0]

    def encode(self, seq):
        encoded_seq = []
        for unit in seq:
            if unit in self.word2index:
                encoded_seq.append(self.word2index[unit])
            else:
                encoded_seq.append(self.word2index['<unk>'])
        return encoded_seq
    
    from torch.utils.data import DataLoader
import torch

def collate_fn(batch, pad_id=0):
    # Lấy max_seq_sample và max_target_sample trong batch
    max_seq_sample = max([item["features_len"] for item in batch])
    max_target_sample = max([item["targets_len"] for item in batch])

    # Khởi tạo tensor padding
    features_padded = []
    targets_padded = []
    features_lens = []
    targets_lens = []

    for item in batch:
        # Pad features
        feature = item["features"]
        pad_length = max_seq_sample - feature.shape[0]
        padded_feature = np.pad(
            feature,
            ((0, pad_length), (0, 0)),
            mode="constant",
            constant_values=0.0
        )
        features_padded.append(padded_feature)
        features_lens.append(item["features_len"])

        # Pad targets
        target = item["targets"]
        pad_length_target = max_target_sample - target.shape[0]
        padded_target = np.pad(
            target,
            (0, pad_length_target),
            mode="constant",
            constant_values=pad_id
        )
        targets_padded.append(padded_target)
        targets_lens.append(item["targets_len"])

    # Chuyển sang tensor
    features_padded = torch.tensor(np.array(features_padded), dtype=torch.float32)
    targets_padded = torch.tensor(np.array(targets_padded), dtype=torch.long)
    features_lens = torch.tensor(features_lens, dtype=torch.long)
    targets_lens = torch.tensor(targets_lens, dtype=torch.long)

    return features_padded, features_lens, targets_padded, targets_lens