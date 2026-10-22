import random
import torch
import torchaudio
from torchaudio import transforms
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset, random_split
import os


class AudioUtil():
    # all the data augmentation
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # random select the max_len
            cut_start = random.randint(0, sig_len-max_len)
            sig = sig[:, cut_start:max_len+cut_start]
        elif sig_len < max_len:
            # length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    # ---
    # raw audio augmentation
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def pitch_shift(aud, shift_limit=[-4, 4]):
        sig, sr = aud
        pitch_shift = np.random.randint(shift_limit[0], shift_limit[1] + 1)
        sig_new = librosa.effects.pitch_shift(sig.numpy(), sr=sr, n_steps=pitch_shift)
        return (torch.Tensor(sig_new), sr)

    @staticmethod
    def time_stretch(aud, shift_limit=[0.9, 1.2]):
        sig, sr = aud
        stretch_time = random.uniform(shift_limit[0], shift_limit[1])
        sig_new = librosa.effects.time_stretch(sig.numpy(), rate=stretch_time)
        return (torch.Tensor(sig_new), sr)

    @staticmethod
    def add_noise(aud):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        wn = np.random.randn(sig_len)
        sig = sig + 0.005*wn
        return (sig, sr)

    @staticmethod
    def add_noise_snr(aud, snr=15, p=0.5):
        aug_apply = torch.distributions.Bernoulli(p).sample()
        if aug_apply.to(torch.bool):
            sig, sr = aud
            num_rows, sig_len = sig.shape
            p_sig = np.sum(abs(sig.numpy())**2)/sig_len
            p_noise = p_sig / 10 ** (snr/10)
            wn = np.random.randn(sig_len) * np.sqrt(p_noise)
            wn_new = np.tile(wn, (num_rows, 1))
            sig_new = sig.numpy() + wn_new
            sig_new = torch.Tensor(sig_new)
        else:
            sig_new, sr = aud
        return (sig_new, sr)

    @staticmethod
    def get_zrc(aud):
        sig, sr = aud
        _, sig_len = sig.shape
        # sig_len_act = np.nonzero(sig.numpy())[-1]
        zcr = librosa.zero_crossings(sig.numpy())
        # zcr_ratio = np.sum(zcr)/sig_len
        zcr = librosa.feature.zero_crossing_rate(sig.numpy(), frame_length=4000, hop_length=2000)
        zcr_mean = np.mean(np.squeeze(zcr))
        zcr_std = np.std(np.squeeze(zcr))
        return zcr_mean, zcr_std

    @staticmethod
    def get_spec(aud):
        sig, sr = aud
        spectral_centroids = librosa.feature.spectral_centroid(sig.numpy(), sr=4000, n_fft=200, hop_length=100)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=sig.numpy(), sr=4000, n_fft=200, hop_length=100)
        return np.mean(spectral_centroids) / 2000, np.std(spectral_centroids)/2000, np.mean(spec_bw)/2000, np.std(spec_bw)/2000

    # -------
    # generate a spectrogram
    # ----
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=512, win_len=None, hop_len=None):
        sig, sr = aud
        top_db = 80

        win_length = int(round(win_len * sr / 1000))
        hop_length = int(round(hop_len * sr / 1000))
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, win_length=win_length,
                                         hop_length=hop_length, n_mels=n_mels, f_min=25, f_max=2000)(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    # ----
    # Augment the spectrogram

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1, p=0.7):
        aug_apply = torch.distributions.Bernoulli(p).sample()
        if aug_apply.to(torch.bool):
            _, n_mels, n_steps = spec.shape
            mask_value = spec.mean()
            aug_spec = spec

            freq_mask_param = max_mask_pct * n_mels

            for _ in range(n_freq_masks):
                aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

            time_mask_param = max_mask_pct * n_steps
            for _ in range(n_time_masks):
                aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec)
        else:
            aug_spec = spec
        return aug_spec


def hand_fea(aud):
    sig, sr = aud
    zcr_mean, zcr_std = AudioUtil.get_zrc(aud)
    spec_mean, spec_std, spbw_mean, spbw_std = AudioUtil.get_spec(aud)
    fea = np.asarray([zcr_mean, zcr_std, spec_mean, spec_std, spbw_mean, spbw_std])
    return fea


class SoundDS(Dataset):
    def __init__(self, df, data_path, mode, df_wide=None):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 15000  # set the duration as 15 s at first
        self.dur = 15
        self.sr = 4000
        self.channel = 1
        self.shift_pct = 0.4
        self.snr = 15
        self.shift_limit = [-4, 4]
        self.stretch_limit = [0.9, 1.2]
        self.mode = mode
        self.df_wide = df_wide

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = os.path.join(self.data_path, self.df.iloc[idx, 1])
        # Get the Class ID
        class_id = self.df.iloc[idx, 2]

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        new_audio = AudioUtil.resample(aud, self.sr)
        # rechan = AudioUtil.rechannel(reaud, self.channel)
        # raw audio augmentation
        if self.mode == 'train':
            # new_audio = AudioUtil.time_stretch(new_audio, self.stretch_limit)
            # new_audio = AudioUtil.pitch_shift(new_audio, self.shift_limit)
            new_audio = AudioUtil.add_noise_snr(new_audio, self.snr, p=0.5)

        new_audio = AudioUtil.pad_trunc(new_audio, self.duration)

        # get the zero-crossing rate
        if aud[0].shape[1] < self.dur * self.sr:  # avoid smaller zcr_ratio caused by the padding
            # zcr_ratio, zcr_std = AudioUtil.get_zrc(aud)
            fea = hand_fea(aud)
        else:
            # zcr_ratio, zcr_std = AudioUtil.get_zrc(new_audio)
            fea = hand_fea(new_audio)
        # get wide features
        # wide_fea_temp = np.append(wide_fea, zcr_ratio)
        # wide_fea_all = torch.Tensor(np.append(wide_fea_temp, zcr_std))
        wide_fea = self.df_wide.iloc[idx, :].values
        wide_fea_all = torch.Tensor(np.append(wide_fea, fea))
        sgram = AudioUtil.spectro_gram(new_audio, n_mels=128, n_fft=512, win_len=50, hop_len=25)

        if self.mode == 'train':
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2, p=0.5)

        return (sgram, wide_fea_all, class_id)


class SoundDS_val(Dataset):
    def __init__(self, recordings,  patch=False):
        self.recordings = recordings
        self.duration = 15000  # set the duration as 15 s at first
        self.sr = 4000
        self.channel = 1
        self.shift_pct = 0.4
        self.snr = 15
        self.shift_limit = [-4, 4]
        self.stretch_limit = [0.9, 1.2]
        self.patch = patch

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.recordings)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        recording = np.reshape(self.recordings[idx], (1, -1))
        recording = recording / (2**(16-1))  # convert from wave read to standard torchaudio input
        aud = (torch.tensor(recording.astype(np.float32)), self.sr)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        new_audio = AudioUtil.resample(aud, self.sr)

        new_audio = AudioUtil.pad_trunc(new_audio, self.duration)

        sgram = AudioUtil.spectro_gram(new_audio, n_mels=128, n_fft=512, win_len=50, hop_len=25)

        return (sgram)
        

class SoundDS_valPatch(Dataset):
    def __init__(self, recordings, wide_fea, patch=True):
        self.recordings = recordings
        self.duration = 15000  # set the duration as 15 s at first
        self.sr = 4000
        self.dur = 15
        self.hop_len = 7.5  # set the patch overlapping as 7.5 seconds
        self.channel = 1
        self.shift_pct = 0.4
        self.snr = 15
        self.shift_limit = [-4, 4]
        self.stretch_limit = [0.9, 1.2]
        self.wide_fea = wide_fea
        self.patch = patch

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.recordings)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        recording = np.reshape(self.recordings[idx], (1, -1))
        recording = recording / (2**(16-1))  # convert from wave read to standard torchaudio input
        aud = (torch.tensor(recording.astype(np.float32)), self.sr)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        
        new_audio = AudioUtil.resample(aud, self.sr)
        num_patch = 1
        if self.patch:
            sig_len = new_audio[0].shape[1] / self.sr
            if sig_len < self.dur:
                num_patch = 1
            else:
                num_patch = np.ceil((sig_len-self.dur) / self.hop_len) + 1

            if num_patch == 1:
                new_audio = AudioUtil.pad_trunc(new_audio, self.duration)
                # get the zero-crossing rate
                if aud[0].shape[1] < self.dur * self.sr:  # avoid smaller zcr_ratio caused by the padding
                    # zcr_ratio, zcr_std = AudioUtil.get_zrc(aud)
                    fea = hand_fea(aud)  # avoid smaller zcr_ratio caused by the padding
                else:
                    fea = hand_fea(new_audio)
                    # zcr_ratio, zcr_std = AudioUtil.get_zrc(new_audio)
                # get wide features
                # wide_fea = self.df_wide.iloc[idx, :].values
                # wide_fea_temp = np.append(self.wide_fea, zcr_ratio)
                # wide_fea_all = [torch.Tensor(np.append(wide_fea_temp, zcr_std))]
                wide_fea_all = [torch.Tensor(np.append(self.wide_fea, fea))]
                sgram_temp = AudioUtil.spectro_gram(new_audio, n_mels=128, n_fft=512, win_len=50, hop_len=25)
                # sgram = [torch.cat((sgram_temp, sgram_temp, sgram_temp), axis = 0)]
                # sgram = [torch.tensor(mono_to_color(sgram_temp.numpy()))]
                sgram = [sgram_temp]
            else:
                sgram = []
                wide_fea_all = []
                end_ind = new_audio[0].shape[1]
                sig, sr = new_audio
                for i in range(int(num_patch)):
                    str_ind = int(i * self.sr * self.hop_len)
                    str_end = int(np.min([str_ind + self.dur * self.sr, end_ind]))
                    audio_seg = sig[:, str_ind:str_end]
                    # get the zero-crossing rate
                    # zcr_ratio_temp, zcr_std_temp = AudioUtil.get_zrc((audio_seg,sr))
                    fea = hand_fea((audio_seg, sr))
                    audio_seg = AudioUtil.pad_trunc((audio_seg, sr), self.duration)
                    # get wide features
                    # wide_fea_temp = self.df_wide.iloc[idx, :].values
                    # wide_fea_temp = np.append(self.wide_fea, zcr_ratio_temp)
                    # wide_fea_all_temp = torch.Tensor(np.append(wide_fea_temp, zcr_std_temp))
                    wide_fea_all_temp = torch.Tensor(np.append(self.wide_fea, fea))
                    # wide_fea_all_temp = torch.Tensor(np.append(wide_fea_temp, zcr_ratio_temp))
                    sgram_temp = AudioUtil.spectro_gram(audio_seg, n_mels=128, n_fft=512, win_len=50, hop_len=25)
                    # sgram.append(torch.tensor(mono_to_color(sgram_temp.numpy())))
                    sgram.append(sgram_temp)
                    # sgram.append(torch.cat((sgram_temp, sgram_temp, sgram_temp), axis = 0))
                    wide_fea_all.append(wide_fea_all_temp)

        return (sgram, wide_fea_all)


class SoundDS_Patch(Dataset):
    def __init__(self, df, data_path, df_wide, patch=True):
        self.df = df
        self.df_wide = df_wide
        self.data_path = data_path
        self.duration = 15000
        self.dur = 15 # set the duration as 15 s at first
        self.sr = 4000
        self.hop_len = 7.5  # set the patch overlapping as 10 seconds
        self.channel = 1
        self.shift_pct = 0.4
        self.snr = 15
        self.shift_limit = [-4, 4]
        self.stretch_limit = [0.9, 1.2]
        self.patch = patch

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = os.path.join(self.data_path, self.df.iloc[idx, 1])
        # Get the Class ID
        class_id = self.df.iloc[idx, 2]

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        new_audio = AudioUtil.resample(aud, self.sr)
        num_patch = 1
        if self.patch:
            sig_len = new_audio[0].shape[1] / self.sr
            if sig_len < self.dur:
                num_patch = 1
            else:
                num_patch = np.ceil((sig_len-self.dur) / self.hop_len) + 1

            if num_patch == 1:
                new_audio = AudioUtil.pad_trunc(new_audio, self.duration)
                # get the zero-crossing rate
                if aud[0].shape[1] < self.dur * self.sr:  # avoid smaller zcr_ratio caused by the padding
                    # zcr_ratio, zcr_std = AudioUtil.get_zrc(aud)
                    fea = hand_fea(aud)  # avoid smaller zcr_ratio caused by the padding
                else:
                    # zcr_ratio, zcr_std = AudioUtil.get_zrc(new_audio)
                    fea = hand_fea(new_audio)
                # get wide features
                wide_fea = self.df_wide.iloc[idx, :].values
                # wide_fea_temp = np.append(wide_fea, zcr_ratio)
                # wide_fea_all = [torch.Tensor(np.append(wide_fea_temp, zcr_std))]
                wide_fea_all = [torch.Tensor(np.append(wide_fea, fea))]
                # wide_fea_all = [torch.Tensor(np.append(wide_fea, zcr_ratio, zcr_std))]
                sgram_temp = AudioUtil.spectro_gram(new_audio, n_mels=128, n_fft=512, win_len=50, hop_len=25)
                # sgram = [torch.cat((sgram_temp, sgram_temp, sgram_temp), axis = 0)]
                # sgram = [torch.tensor(mono_to_color(sgram_temp.numpy()))]
                sgram = [sgram_temp]
            else:
                sgram = []
                wide_fea_all = []
                end_ind = new_audio[0].shape[1]
                sig, sr = new_audio
                for i in range(int(num_patch)):
                    str_ind = int(i * self.sr * self.hop_len)
                    str_end = int(np.min([str_ind + self.dur * self.sr, end_ind]))
                    audio_seg = sig[:, str_ind:str_end]
                    # get the zero-crossing rate
                    # zcr_ratio_temp, zcr_std_temp = AudioUtil.get_zrc((audio_seg, sr))
                    fea = hand_fea((audio_seg, sr))
                    audio_seg = AudioUtil.pad_trunc((audio_seg, sr), self.duration)
                    # get wide features
                    wide_fea_temp = self.df_wide.iloc[idx, :].values
                    # wide_fea_temp = np.append(wide_fea_temp, zcr_ratio_temp)
                    # wide_fea_all_temp = torch.Tensor(np.append(wide_fea_temp, zcr_std_temp))
                    wide_fea_all_temp = torch.Tensor(np.append(wide_fea_temp, fea))
                    # wide_fea_all_temp = torch.Tensor(np.append(wide_fea_temp, zcr_ratio_temp))
                    sgram_temp = AudioUtil.spectro_gram(audio_seg, n_mels=128, n_fft=512, win_len=50, hop_len=25)
                    # sgram.append(torch.tensor(mono_to_color(sgram_temp.numpy())))
                    sgram.append(sgram_temp)
                    # sgram.append(torch.cat((sgram_temp, sgram_temp, sgram_temp), axis = 0))
                    wide_fea_all.append(wide_fea_all_temp)

        return (sgram, wide_fea_all, class_id)
