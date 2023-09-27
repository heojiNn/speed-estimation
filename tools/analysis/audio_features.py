import argparse
import contextlib
import math
import os
import os.path as osp
import random
import wave
import audplot

import numpy as np
import matplotlib
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning

import librosa
import librosa.display

from scipy import signal
from scipy.io.wavfile import write

import torch
import torchaudio

import warnings
warnings.filterwarnings(action='ignore', category=MatplotlibDeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

def parse_args():
    """Returns: cl args"""
    parser = argparse.ArgumentParser('KIRun Speed Training')
    parser.add_argument('--loadpath', help='Path to runner data (../kirun-data-by-runner)', required=False)
    parser.add_argument('--savepath', help='Path where results are to be stored', required=False)
    parser.add_argument('--single', help='Analyse single audio file', action='store_true')
    parser.add_argument('--multiple', help='Analyse multiple audio files', action='store_true')
    parser.add_argument('--sessions', action='store_true')
    parser.add_argument('--samplerate', type=int, default=16000)

    args = parser.parse_args()
    return args

########################################################################################################################

# MAIN REFERENCE: librosa-framework
# https://librosa.org/doc/main/feature.html#spectral-features

########################################################################################################################

# PLOT FEATURES:

# WAVEFORM:

def plot_signal(y, sr, seconds, label='', show=False):
    if type(y) is type(list):
        for signal in y:
            plt.plot(signal)
            plt.title('Signal {}'.format(label))
            plt.xlabel('Time (samples), {seconds}s * {sr}Hz'.format(seconds=seconds, sr=sr))
            plt.ylabel('Amplitude')
    else:
        plt.plot(y)
        plt.title('Signal {}'.format(label))
        plt.xlabel('Time (samples), {seconds}s * {sr}Hz'.format(seconds=seconds, sr=sr))
        plt.ylabel('Amplitude')
    if show:
        plt.title(f'Waveform, {sr} Hz')
        plt.show()

# SPECTROGRAM:

def plot_spectrogram(y, sr, show=False):
    run, _ = librosa.effects.trim(y)  # trim silent edges
    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(run, n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    if show:
        plt.title(f'Spectrogram, {sr} Hz')
        # plt.colorbar(format='%+2.0f dB')
        plt.show()

# MEL-SPECTROGRAM, MFCC, CHROMA, ...

def plot_spectral_features(y, sr, path=None, show=False, _type='melspec'):
    """
    https://librosa.org/doc/main/feature.html#spectral-features

    Args:
        _type: 'melspec', 'chroma_stft', 'chroma_cqt', 'mfcc', 'spectral_contrast', 'poly_features', 'tonnetz'
        y: input signal
        sr: sample-rate
        show: Plot
    Returns: None
    """

    # trim silent edges
    run, _ = librosa.effects.trim(y)
    # librosa.display.waveshow(run, sr=sr)
    n_fft = 320  # 2048, 320
    win_length = n_fft
    hop_length = 160

    n_mels = 64
    n_chroma = 12
    # D = np.abs(librosa.stft(run[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    # plt.plot(D)

    zero_pad = True

    if _type == 'melspec':
        S = librosa.feature.melspectrogram(run, sr=sr, center=zero_pad, n_mels=n_mels, win_length=win_length)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'chroma_stft':
        S = librosa.feature.chroma_stft(run, sr=sr, center=zero_pad, n_chroma=n_chroma, win_length=win_length)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'chroma_cqt':
        S = librosa.feature.chroma_cqt(run, sr=sr, n_chroma=n_chroma)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_hz')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'chroma_cens':
        S = librosa.feature.chroma_cens(run, sr=sr, n_chroma=n_chroma, win_length=win_length)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'mfcc':
        n_mfcc = 13
        S = librosa.feature.mfcc(run, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop_length,
                                 n_fft=n_fft, win_length=win_length, center=zero_pad)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'spectral_contrast':
        S = librosa.feature.spectral_contrast(run, sr=sr, center=zero_pad)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'poly_features':
        S = librosa.feature.poly_features(run, sr=sr, center=zero_pad)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='f_bins')
        print(f'{_type} Shape: {S.shape}')
    elif _type == 'tonnetz':
        S = librosa.feature.tonnetz(run, sr=sr, n_chroma=n_chroma)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tonnetz')
        print(f'{_type} Shape: {S.shape}')
    else:
        print(f'Type {_type} not defined.')
        return RuntimeError()

    if show:
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectral Feature: {_type}')
        plt.show()

# ALL IN ONE (HAPPYMEAL):

def plot_all_spectral_features(y, sr, path=None, show=False, _type=None):
    """
    https://librosa.org/doc/main/feature.html#spectral-features

    Args:
        _type: 'melspec', 'chroma_stft', 'chroma_cqt', 'mfcc', 'spectral_contrast', 'poly_features', 'tonnetz'
        y: input signal
        sr: sample-rate
        show: Plot
    Returns: None
    """

    # trim silent edges
    run, _ = librosa.effects.trim(y)
    if path is not None:
        run, _ = read_audio(path, start=0.0, seconds=5.0, _sr=sr, verbose=True)

    # DEFINE SETTINGS:
    n_fft = 320  # 2048, 320
    win_length = n_fft
    hop_length = 160
    n_mels = 64
    n_chroma = 12
    zero_pad = True
    n_mfcc = 13

    if _type is None:
        # Create plots
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(7, 9))

        # Add vertical spacing between plots:
        plt.subplots_adjust(hspace=0.5)

        # WAVEFORM:
        axs[0].set(title='WAVEFORM')
        librosa.display.waveshow(run, sr=sr, ax=axs[0])
        axs[0].set_xlim(0.0, 5.0)
        axs[0].set_ylabel('Amplitude')
        axs[0].title.set_size(7)

        # SPECTROGRAM:
        # axs[0, 1].set(title='SPECTROGRAM')
        # S = librosa.stft(y)
        # S_DB = librosa.power_to_db(abs(S), ref=np.max)
        # librosa.display.specshow(S_DB, sr=sr, ax=axs[0, 1], hop_length=hop_length, x_axis='time', y_axis='mel')

        # MEL-SPEC:
        axs[1].set(title='MEL-SPEC')
        S = librosa.feature.melspectrogram(run, sr=sr, center=zero_pad, n_mels=n_mels, win_length=win_length)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, ax=axs[1], hop_length=hop_length, x_axis='time', y_axis='mel')
        #axs[1].set_xlim(0.0, 5.0)
        axs[1].title.set_size(7)

        # MFCC:
        axs[2].set(title='MFCC')
        S = librosa.feature.mfcc(run, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop_length,
                                 n_fft=n_fft, win_length=win_length, center=zero_pad)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, ax=axs[2], hop_length=hop_length, x_axis='time', y_axis='mel')
        #axs[2].set_xlim(0.0, 5.0)
        axs[2].title.set_size(7)

        # TONNETZ:
        # axs[3, 1].set(title='TONNETZ')
        # S = librosa.feature.tonnetz(run, sr=sr, n_chroma=n_chroma)
        # S_DB = librosa.power_to_db(S, ref=np.max)
        # librosa.display.specshow(S_DB, sr=sr, ax=axs[3, 1], hop_length=hop_length, x_axis='time', y_axis='tonnetz')

        # CHROMA_STFT:
        # axs[2, 0].set(title='CHROMA_STFT')
        # S = librosa.feature.chroma_stft(run, sr=sr, center=zero_pad, n_chroma=n_chroma, win_length=win_length)
        # S_DB = librosa.power_to_db(S, ref=np.max)
        # librosa.display.specshow(S_DB, sr=sr, ax=axs[2, 0], hop_length=hop_length, x_axis='time', y_axis='chroma')

        # CHROMA_CQT:
        axs[3].set(title='CHROMA_CQT')
        S = librosa.feature.chroma_cqt(run, sr=sr, n_chroma=n_chroma)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, ax=axs[3], hop_length=hop_length, x_axis='time', y_axis='cqt_hz')
        #axs[3].set_xlim(0.0, 5.0)
        axs[3].title.set_size(7)

        # CHROMA_CENS:
        # axs[1, 1].set(title='CHROMA_CENS')
        # S = librosa.feature.chroma_cens(run, sr=sr, n_chroma=n_chroma)
        # S_DB = librosa.power_to_db(S, ref=np.max)
        # librosa.display.specshow(S_DB, sr=sr, ax=axs[1, 1], hop_length=hop_length, x_axis='time', y_axis='chroma')

        # # SPECTRAL CONTRAST:
        # axs[1, 3].set(title='SPECTRAL CONTRAST')
        # S = librosa.feature.spectral_contrast(run, sr=sr, center=zero_pad)
        # S_DB = librosa.power_to_db(S, ref=np.max)
        # librosa.display.specshow(S_DB, sr=sr, ax=axs[1, 3], hop_length=hop_length, x_axis='time', y_axis='chroma')

    new = True
    if new:
        # Load the waveform and compute the Mel-spectrogram
        waveform = y

        # Plot the Multiview display
        plt.figure(figsize=(10, 11))

        ########
        # Plot the waveform
        plt.subplot(4, 1, 1)
        librosa.display.waveshow(waveform, sr=sr)
        #plt.title.set_size(7)
        plt.title('Waveform (16kHz)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Amplitude', fontsize=14)

        plt.xlim(0.0, 5.0)
        xticks = [0, 0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2, 4.8]
        plt.xticks(xticks)
        plt.gca().xaxis.set_label_coords(.98, -0.1)

        ########
        # Plot the Mel-spectrogram
        plt.subplot(4, 1, 2)
        #
        mel_spec = librosa.feature.melspectrogram(waveform, sr=sr)
        librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
        #plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-spectrogram (b=64)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Hz', fontsize=14)
        plt.gca().xaxis.set_label_coords(.98, -0.1)

        ########
        # Plot the MFCCs
        plt.subplot(4, 1, 3)
        #
        S = librosa.feature.mfcc(run, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop_length,
                                 n_fft=n_fft, win_length=win_length, center=zero_pad)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=hop_length,
                                 x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB')
        plt.title('MFCCs (b=13)', fontsize=13)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Hz', fontsize=14)
        plt.gca().xaxis.set_label_coords(.98, -0.1)

        ########
        # Plot the CHROMA_STFT
        plt.subplot(4, 1, 4)

        # S = librosa.feature.chroma_stft(run, sr=sr, center=zero_pad, n_chroma=n_chroma, win_length=win_length)
        # S_DB = librosa.power_to_db(S, ref=np.max)
        # librosa.display.specshow(S_DB, sr=sr, ax=axs[2, 0], hop_length=hop_length, x_axis='time', y_axis='chroma')
        S = librosa.feature.chroma_stft(run, sr=sr, center=zero_pad, n_chroma=n_chroma, win_length=win_length)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='chroma')
        plt.title('CHROMA_STFT, (b=12)', fontsize=13)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Pitch class', fontsize=14)
        plt.gca().xaxis.set_label_coords(.98, -0.1)

        #
        # S = librosa.feature.chroma_cqt(run, sr=sr, n_chroma=n_chroma)
        # librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_hz')
        # # plt.colorbar(format='%+2.0f dB')
        # plt.title('CHROMA_STFT', fontsize=9)
        # plt.xlabel('Time (s)', fontsize=7)

        # Adjust the y-ticks
        # N = 20  # 1 tick every 3
        # yticks_pos, yticks_labels = plt.yticks()  # get all axis ticks
        # myticks = [j for i, j in enumerate(yticks_pos) if not i % N]  # index of selected ticks
        # newlabels = [label for i, label in enumerate(yticks_labels) if not i % N]
        # plt.gca().set_yticks(myticks)  # set new X axis ticks

        # Adjust the y-ticks
        #num_y_ticks = 10  # Specify the desired number of y-ticks
        #y_ticks = np.linspace(0, sr / 2, num_y_ticks)  # Generate evenly spaced y-tick values
        # num_y_ticks = 6
        # y_ticks = np.linspace(32, 62, num_y_ticks)
        # plt.yticks(y_ticks)


        ########
        # Adjust the vertical spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(fname='tools/analysis/features1.pdf', format='pdf')
        plt.show()

    show = False
    if show:
        # fig.colorbar(format='%+2.0f dB')
        # plt.title(f'Spectral Feature: {_type}')

        # axes = plt.gca()
        # axes.title.set_size(5)

        plt.savefig(fname='tools/analysis/features2.pdf', format='pdf')
        fig.show()

# STFT, HANN & FTT:

def stft_window_function(y, sr, dur, count=5):
    """ This function visualizes how stft functions"""
    # define window function
    window_length = 320
    overlap = window_length // 2
    window = signal.windows.hann(window_length)
    plt.plot(window)
    plt.title("Hann window")
    plt.ylabel("Amplitude")
    plt.xlabel("Window length")
    plt.show()

    # window audio
    start = 0 * overlap
    end = window_length
    windowed = []
    for i in range(1, count):
        print(f'start: {start}, end: {end}')
        assert (i * window_length) - overlap < sr * dur, 'tring to window non-existant audio'
        windowed_segment = y[start:end] * window
        windowed.append(windowed_segment)
        plt.plot(windowed_segment)
        # plt.yscale('log')
        plt.title(f"Window {i}")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()
        # next window
        start = i * overlap
        end = start + window_length
    # apply fft to each windowed audio and average to one spectrogram:
    for i, w in enumerate(windowed):
        fft = librosa.stft(w)
        fft_db = librosa.amplitude_to_db(abs(fft))
        # Plot the spectrogram
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(fft_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

########################################################################################################################

# UTIL:

def read_audio(loadpath, start=0.0, seconds=5.0, _sr=None, verbose=False):
    if seconds != 0.0:
        if _sr is None:
            y, sr = librosa.load(loadpath, offset=start, duration=seconds)
        else:
            y, sr = librosa.load(loadpath, offset=start, duration=seconds, sr=_sr)
        if verbose:
            print(f'Reading {seconds} Seconds. Start @ {start}.')
    else:  # read total
        if _sr is None:
            y, sr = librosa.load(loadpath, offset=start)
        else:
            y, sr = librosa.load(loadpath, offset=start, sr=_sr)
        if verbose:
            print(f'Reading total audio. Start @ {start}.')
    return y, sr


def write_audio(savepath, y, sr):
    write(savepath, sr, y)


def get_wav_duration(wav_path):
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def amplitude_to_db(waveform):
    db = librosa.amplitude_to_db(np.abs(waveform))
    return db


def db_to_amplitude(db_waveform):
    amplitude = librosa.db_to_amplitude(db_waveform)
    return amplitude

########################################################################################################################


def main():
    args = parse_args()
    print('\n', '=' * 15, 'KIRun Audio ', '=' * 15)

    start = 45.0
    dur = 5.0
    end = start + dur

    # ARGS
    single = args.single
    multiple = args.multiple
    session = args.sessions
    loadpath = args.loadpath
    savepath = args.savepath
    SAMPLE_RATE = args.samplerate

    # spectral_features = ['melspec']
    spectral_features = ['melspec', 'chroma_stft', 'chroma_cqt', 'mfcc', 'spectral_contrast', 'tonnetz']
    # spectral_features = ['melspec', 'chroma_stft', 'chroma_cqt', 'mfcc', 'spectral_contrast', 'poly_features', 'tonnetz']

    # Analyse single soundfile:
    if single:
        print('Analysing single .wav file:')
        wv = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\kirun-data-by-runner\viS8m4k5BDY2YoAhILBdFkgpJgy1\s96sa\-Mfh5YB8IqzYIgZHCbrP\-Mfh5YB8IqzYIgZHCbrP___2021-07-28T12-32-35-208Z.wav'
        y, sr = read_audio(wv, start=start, seconds=dur, _sr=SAMPLE_RATE, verbose=True)
        # plot_signal(y, sr, seconds=dur, show=True)  # amplitude

        # stft_window_function(y, sr=sr, dur=dur)
        # time = librosa.get_duration(y, SAMPLE_RATE)
        # plot_spectrogram(y, SAMPLE_RATE, show=True)
        # # stretch = torchaudio.transforms.TimeStretch()
        # for feature in spectral_features:
        #     plot_spectral_features(y, SAMPLE_RATE, path=wv, show=True, _type=feature)

        plot_all_spectral_features(y, SAMPLE_RATE, show=True)

        # Create plots
        #fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(6, 4))

        # audplot.signal(y, sr)
        # plt.show()

        # y_db = librosa.power_to_db(y, ref=np.max)
        # hop_dur = 160  # default hop length is 512
        # centers = librosa.mel_frequencies(n_mels=64, fmax=8000)
        # image = audplot.spectrum(y_db, hop_dur, centers)
        # cb = plt.colorbar(image, format='%+2.0f dB')
        # cb.outline.set_visible(False)
        # plt.tight_layout()
        # plt.show()



        # zerocrossings = librosa.zero_crossings(y)
        # print(zerocrossings.shape)


    # Analyse multiple soundfiles:
    if multiple:
        assert loadpath is not None, 'Please specify a loadpath for multiple audio analysis'

        print('Analysing mulitple .wav files:')
        dirs = os.listdir(loadpath)
        print('Number of session-dirs:', len(dirs))
        print('session-dir-names:', dirs)

        count = 0
        sessions = []
        for ses in dirs:
            path = osp.join(loadpath, ses)
            count += len(os.listdir(path))
            sessions.extend(os.listdir(path))
        # print('Number of runs in total:', len(sessions), count) # doesnt account for subfolders
        print('Number of unique runners:', len(set(sessions)))

        # multiple runs of ONE (toplevel) session dir:
        if session:
            print('Reading session-dependant .wav files:')
            # pick random dir:
            rand = random.randint(0, len(dirs)-1)
            print('Selected Session:', dirs[rand])
            session_runner = os.listdir(osp.join(loadpath, dirs[rand]))
            print('Session Runners:', session_runner)

            wav_paths = []
            session_path = osp.join(loadpath, dirs[rand])  # i.e. viS8m4k5BDY2YoAhILBdFkgpJgy1

            # iterate through all runners .wav files and save the paths:
            print('Reading .wav files:')
            for runner in session_runner:  # i.e. ['b61kh', 'b75ps', 'b86ke', 'b97fm', ...]
                runner_path = osp.join(session_path, runner)
                print(runner_path)

                for run in os.listdir(runner_path):  # run ~= '-M....'
                    run_path = osp.join(runner_path, run)
                    print(run_path)

                    for run_file in os.listdir(run_path):  # run_files: .json, .csv, .gpx, .txt, .wav
                        if run_file.endswith('.wav'):
                            wav_path = osp.join(run_path, run_file)
                            wav_paths.append(wav_path)
                            break

            print('Finished Reading .wav files!')
            print('Number of .wav-files:', len(wav_paths))

            # ----------------------------------------------------------------------------------------------------------
            # Adjust Plotting Arguments:

            start = 0  # 45
            seconds = 180.0
            num_read = 10

            read_end = False
            limit_read = True
            plotsignal = False
            plotspectogram = False
            plotmelspectogram = True

            plot_lbl = ''

            signals = []

            if limit_read:
                print(f'Limiting number of samples to load, from {len(wav_paths)} to {num_read}.')
                wav_paths = random.sample(wav_paths, num_read)

            # load .wav files:
            for wav in wav_paths:

                # watch end of signal
                if read_end:

                    duration = get_wav_duration(wav)
                    start = duration - seconds
                    print(f'Reading signal end. Duration: {duration}, Start: {start}')
                    signal = read_audio(wav, start=start, seconds=duration, _sr=SAMPLE_RATE)[0]

                    if plotsignal:
                        # plot single signal:
                        plot_signal(signal, sr=SAMPLE_RATE, seconds=seconds, label=plot_lbl)
                    elif plotspectogram:
                        # plot spectogram
                        plot_spectrogram(signal, SAMPLE_RATE)
                    elif plotmelspectogram:
                        # plot spectogram
                        plot_spectral_features(signal, SAMPLE_RATE)

                # watch start of signal
                else:
                    signal = read_audio(wav, start=start, seconds=seconds, _sr=SAMPLE_RATE)[0]
                    if plotsignal:
                        # plot single signal:
                        plot_signal(signal, sr=SAMPLE_RATE, seconds=seconds, label=plot_lbl)
                    elif plotspectogram:
                        # plot spectogram
                        plot_spectrogram(signal, SAMPLE_RATE)
                    elif plotmelspectogram:
                        # plot spectogram
                        plot_spectral_features(signal, SAMPLE_RATE)

                signals.append(signal)

            plt.show()


if __name__ == '__main__':
    main()




