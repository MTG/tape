import torch
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn.init as init
import librosa
import sys, math
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import time
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data import DataLoader
from mir_eval import melody
from scipy.stats import norm
import pandas as pd
import argparse
from librosa.sequence import viterbi_discriminative
from scipy.ndimage import gaussian_filter1d
import json


class PitchEstimator(nn.Module):
    def __init__(self, labeling, sr=16000, window_size=1024, hop_length=160):
        super().__init__()
        self.labeling = labeling
        self.sr = sr
        self.window_size = window_size
        self.hop_length = hop_length

    def estimate(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)  # separate from forward since we used BCEWithLogitsLoss
        return x

    def get_activation(self, audio, center=True, batch_size=128):
        """
        audio : (N,) only accept mono audio with a specific sampling rate
        """
        assert len(audio.shape) == 1

        def get_frame(audio, center):
            if center:
                audio = nn.functional.pad(audio, pad=(self.window_size // 2, self.window_size // 2))
            # make 1024-sample frames of the audio with hop length of 10 milliseconds
            n_frames = 1 + int((len(audio) - self.window_size) / self.hop_length)
            assert audio.dtype == torch.float32
            itemsize = 1  # float32 byte size
            frames = torch.as_strided(audio, size=(self.window_size, n_frames),
                                      stride=(itemsize, self.hop_length * itemsize))
            frames = frames.transpose(0, 1).clone()

            frames -= (torch.mean(frames, axis=1).unsqueeze(-1))
            frames /= (torch.std(frames, axis=1).unsqueeze(-1))
            return frames

        frames = get_frame(audio, center)
        activation_stack = []
        device = self.linear.weight.device

        for i in range(0, len(frames), batch_size):
            f = frames[i:min(i + batch_size, len(frames))]
            f = f.to(device)
            act = self.estimate(f)
            activation_stack.append(act.cpu())
        activation = torch.cat(activation_stack, dim=0)
        return activation

    # todo: move to tensor. currently returns numpy
    def predict(self, audio, viterbi=False, center=True, batch_size=128):
        self.eval()
        with torch.no_grad():
            activation = self.get_activation(audio, batch_size=batch_size)
            frequency = self.to_freq(activation, viterbi=viterbi)
            confidence = activation.max(dim=1)[0]
            t = torch.arange(confidence.shape[0]) * self.hop_length / self.sr
            return t.numpy(), frequency, confidence.numpy(), activation.numpy()

    def process_file(self, file, output=None, viterbi=False,
                     center=True, save_plot=False, batch_size=128):
        audio, _ = librosa.load(file, sr=self.sr, mono=True)
        audio = torch.from_numpy(audio)
        t, frequency, confidence, activation = self.predict(
            audio,
            viterbi=viterbi,
            center=center,
            batch_size=batch_size,
        )

        f0_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + ".f0.csv"
        f0_data = np.vstack([t, frequency, confidence]).transpose()
        np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f'], delimiter=',',
                   header='time,frequency,confidence', comments='')

        # save the salience visualization in a PNG file
        if save_plot:
            import matplotlib.cm
            from imageio import imwrite

            plot_file = os.path.join(output, os.path.basename(os.path.splitext(file)[0])) + ".activation.png"
            # to draw the low pitches in the bottom
            salience = np.flip(activation, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())

            imwrite(plot_file, (255 * image).astype(np.uint8))

    # todo: currently in numpy. move to tensor
    def to_local_average_cents(self, salience, center=None):
        """
        find the weighted average cents near the argmax bin
        """
        return self.labeling.label2c(salience, center=center)

    # todo: currently in numpy. move to tensor
    def to_viterbi_cents(self, salience):
        """
        Find the Viterbi path using a transition prior that induces pitch
        continuity.
        """
        # transition probabilities inducing continuous pitch
        # big changes are penalized with one order of magnitude
        transition = gaussian_filter1d(np.eye(self.labeling.n_bins), 30) + 99 * gaussian_filter1d(
            np.eye(self.labeling.n_bins), 2)
        transition = transition / np.sum(transition, axis=1)[:, None]

        p = salience / salience.sum(axis=1)[:, None]
        p[np.isnan(p.sum(axis=1)), :] = np.ones(self.labeling.n_bins) * 1 / self.labeling.n_bins
        path = viterbi_discriminative(p.T, transition)

        return path, np.array([self.to_local_average_cents(salience[i, :], path[i]) for i in
                               range(len(path))])

    def to_freq(self, activation, viterbi=False):
        if viterbi:
            path, cents = self.to_viterbi_cents(activation.detach().numpy())
        else:
            cents = self.to_local_average_cents(activation.detach().numpy())

        # cents = torch.tensor(cents) # todo: all computations should take tensor
        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0
        # frequency[torch.isnan(frequency)] = 0
        return frequency


class Label:
    def __init__(self, n_bins=360, min_f0_hz=31.70,
                 granularity_c=20, smooth_std_c=25):
        self.n_bins = n_bins
        self.min_f0_hz = min_f0_hz
        self.min_f0_c = melody.hz2cents(np.array([min_f0_hz]))[0]
        self.granularity_c = granularity_c
        self.smooth_std_c = smooth_std_c
        self.pdf_normalizer = norm.pdf(0)
        self.centers_c = np.linspace(0, (self.n_bins - 1) * self.granularity_c, self.n_bins) + self.min_f0_c
        self.centers_hz = 10 * 2 ** (self.centers_c / 1200)

    def c2label(self, pitch_c):
        result = norm.pdf((self.centers_c - pitch_c) / self.smooth_std_c).astype(np.float32)
        result /= self.pdf_normalizer
        return result

    def hz2label(self, pitch_hz):
        pitch_c = melody.hz2cents(np.array([pitch_hz]))[0]
        return self.c2label(pitch_c)

    def label2c(self, salience, center=None):
        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(salience * self.centers_c[start:end])
            weight_sum = np.sum(salience)
            return product_sum / np.clip(weight_sum, 1e-8, None)
        if salience.ndim == 2:
            return np.array([self.label2c(salience[i, :]) for i in range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")

    def label2hz(self, salience):
        return 10 * 2 ** (self.label2c(salience) / 1200)


class ConvBlock(nn.Module):
    def __init__(self, f, w, s, d, in_channels):
        super().__init__()
        p1 = d * (w - 1) // 2
        p2 = d * (w - 1) - p1
        self.pad = nn.ZeroPad2d((0, 0, p1, p2))

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=(s, 1),
                                dilation=(d, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CREPE(PitchEstimator):
    def __init__(self, labeling, sr=16000, window_size=1024, hop_length=160, model_capacity="full"):
        super().__init__(labeling, sr=sr, window_size=window_size, hop_length=hop_length)

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]
        self.labeling = labeling
        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [4, int(window_size // 1024), 1, 1, 1, 1]
        dilations = [1, 1, 1, 1, 1, 1]

        for i in range(len(self.layers)):
            f, w, s, d, in_channel = filters[i + 1], widths[i], strides[i], dilations[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, d, in_channel))

        self.linear = nn.Linear(64 * capacity_multiplier, self.labeling.n_bins)
        self.eval()

    def forward(self, x):
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class Pathway(nn.Module):
    def __init__(self, window_size=1024, model_capacity="full", n_layers=6):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]
        self.layers = [1, 2, 3, 4, 5, 6]
        self.layers = self.layers[:n_layers]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [4, 1, 1, 1, 1, 1]
        total_dilation = int(np.log2(window_size / 1024))
        dilations = [2 for dilation in range(total_dilation)] + [1 for no_dilation in range(6 - total_dilation)]
        strides = [s * dilations[i] for i, s in enumerate(strides)]

        for i in range(len(self.layers)):
            f, w, s, d, in_channel = filters[i + 1], widths[i], strides[i], dilations[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, d, in_channel))

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)
        x = x.permute(0, 3, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :])
        return self.dropout(x)


class TwoStreams(PitchEstimator):
    def __init__(self, labeling, sr=16000, window_size=1024, hop_length=160, model_capacity="full", nhead=8):
        super().__init__(labeling, sr=sr, window_size=window_size, hop_length=hop_length)

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]
        self.labeling = labeling

        self.slow = Pathway(window_size, model_capacity)
        self.fast = Pathway(1024, model_capacity)

        self.pe = PositionalEncoding(d_model=16 * capacity_multiplier)
        self.encoder1 = nn.TransformerEncoderLayer(
            d_model=16 * capacity_multiplier, nhead=nhead, batch_first=True, dropout=0.25)
        self.encoder2 = nn.TransformerEncoderLayer(
            d_model=16 * capacity_multiplier, nhead=nhead, batch_first=True, dropout=0.25)
        self.decoder1 = nn.TransformerDecoderLayer(
            d_model=16 * capacity_multiplier, nhead=nhead, batch_first=True, dropout=0.25)
        self.decoder2 = nn.TransformerDecoderLayer(
            d_model=16 * capacity_multiplier, nhead=nhead, batch_first=True, dropout=0.25)

        self.linear = nn.Linear(64 * capacity_multiplier, self.labeling.n_bins)

        self.eval()

    def forward(self, x):
        # x : shape (batch, sample)
        x_slow = self.slow(x)
        x_slow = self.pe(x_slow.squeeze(1))
        x_slow = self.encoder1(x_slow)
        x_slow = self.encoder2(x_slow)

        center = self.window_size // 2
        x_fast = self.fast(x[:, center - 512:center + 512])
        x_fast = self.pe(x_fast.squeeze(1))
        x_fast = self.decoder1(x_fast, x_slow)
        x_fast = self.decoder2(x_fast, x_slow)

        x = x_fast.unsqueeze(1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class TAPE(TwoStreams):
    def __init__(self, instrument='violin', window_size=None, hop_length=None):
        assert instrument == 'violin', 'As of now, the only supported instrument is the violin'
        package_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(package_dir, instrument, instrument + "_range.json"), "r") as f:
            args = json.load(f)
        labeling = Label(n_bins=args['instrument_n_bins'], min_f0_hz=args['instrument_min_hz'],
                         granularity_c=args['instrument_granularity_c'], smooth_std_c=args['instrument_smooth_std_c'])
        if not window_size:
            window_size = args['window_size']
        if not hop_length:
            hop_length = args['hop_length']
        super().__init__(labeling, sr=args['sampling_rate'], window_size=window_size, hop_length=hop_length)
        self.model_url = args['model_file']
        self.load_weight(instrument)
        self.eval()

    def load_weight(self, instrument):
        self.download_weights(instrument)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "{}.pt".format(instrument)
        self.load_state_dict(torch.load(os.path.join(package_dir, instrument, filename)))

    def download_weights(self, instrument):
        weight_file = "{}.pt".format(instrument)

        # in all other cases, decompress the weights file if necessary
        package_dir = os.path.dirname(os.path.realpath(__file__))
        weight_path = os.path.join(package_dir, instrument, weight_file)
        if not os.path.isfile(weight_path):
            try:
                from urllib.request import urlretrieve
            except ImportError:
                from urllib import urlretrieve
            print('Downloading weight file {} from {} ...'.format(weight_path, self.model_url))
            urlretrieve(self.model_url, weight_path)
