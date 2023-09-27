import math
import os
import os.path as osp
import sys
import warnings
import wave
import audmetric
import audobject
import audtorch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchaudio
import tqdm
from matplotlib import MatplotlibDeprecationWarning
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LinearRegression
import seaborn as sns
import re
import audiofile as af
import logging
from scipy.signal import butter, filtfilt


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MinMaxScaler(audobject.Object):
    def __init__(self, minimum: float, maximum: float):
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def encode(self, x):
        return (x - self.minimum) / (self.maximum - self.minimum)

    def decode(self, x):
        return x * (self.maximum - self.minimum) + self.minimum


class Config:
    """Contains Hyperparameters."""

    def __init__(self, window_size, fold, epochs, optimizer, batch_size, learning_rate, approach):
        self.window_size = window_size
        self.fold = fold
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.approach = approach

    def __repr__(self):
        return str(
            f"wdw:{self.window_size} " + f"fold:{self.fold} " + f"epochs:{self.epochs} " + f"opt:{self.optimizer} " + f"b_size:{self.batch_size} " + f"lr:{self.learning_rate} " + f"{self.approach}")


# ###################
# eval util
# ###################


def init_logger(log_dir, filemode):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def transfer_features(features, device):
    return features.to(device).float()


def evaluate_regression(model, device, loader, transfer_func, scaler, approach, integer_base=False):
    metrics = {
        'CC': audmetric.pearson_cc,
        'CCC': audmetric.concordance_cc,
        'MSE': audmetric.mean_squared_error,
        'MAE': audmetric.mean_absolute_error
    }

    model.to(device)
    model.eval()

    outputs = torch.zeros(len(loader.dataset))
    targets = torch.zeros(len(loader.dataset))

    val_loss = []
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for index, (features, target) in tqdm.tqdm(
                enumerate(loader),
                desc='Batch',
                total=len(loader)
        ):
            start_index = index * loader.batch_size
            end_index = (index + 1) * loader.batch_size
            if end_index > len(loader.dataset):
                end_index = len(loader.dataset)

            if approach in ['ast', 'passt', 'wvcnn14', 'wvlmcnn14', 'wvlmcnn14lstm', 'hts', 'specnn']:
                x = transfer_features(features, device).squeeze(1)
                outputs[start_index:end_index] = model(x).squeeze(1)
                targets[start_index:end_index] = target
            else:
                outputs[start_index:end_index] = model(transfer_features(features, device)).squeeze(1)
                targets[start_index:end_index] = target

    loss = criterion(outputs, targets.float())
    if approach == 'ast':
        loss = criterion(outputs, targets.half())
    val_loss.append(loss.item())
    targets = targets.numpy()
    outputs = outputs.cpu().numpy()

    if integer_base:
        # integer-based == round to closest lying integer to compare with labels.
        outputs = np.round(outputs)

    if scaler is not None:
        targets = scaler.decode(targets)
        outputs = scaler.decode(outputs)

    results = {
        key: metrics[key](targets, outputs) for key in metrics.keys()
    }
    return results, targets, outputs, val_loss


def evaluate_sed(model, device, loader, approach, nclasses, eval=False):
    metrics = {
        'ACC': audmetric.accuracy,
        'PRECISION': precision_score,
        'F1': f1_score,
        'RECALL': recall_score
    }

    model.to(device)
    model.eval()

    outputs = torch.zeros(len(loader.dataset) * nclasses)
    targets = torch.zeros(len(loader.dataset) * nclasses)

    # TODO: fix this workaround for multidimensional
    outputs2 = []
    targets2 = []


    val_loss = []
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for index, (features, target) in tqdm.tqdm(
                enumerate(loader),
                desc='Batch',
                total=len(loader)
        ):
            start_index = index * loader.batch_size * nclasses
            end_index = (index + 1) * loader.batch_size * nclasses

            if end_index > len(loader.dataset) * nclasses:
                end_index = len(loader.dataset) * nclasses

            if approach in ['ast', 'passt', 'wvcnn14', 'wvlmcnn14', 'wvlmcnn14lstm', 'specnn']:
                x = transfer_features(features, device).squeeze(1)
            else:
                x = transfer_features(features, device)

            # model output: torch.Size([B, Window-size * Segment-size]), e.g [32, 5000ms/200ms] = [32, 25]
            # targets should also be: [B, 25] (in datasets.py)
            # forward batch:
            output_tensor = model(x).squeeze(1)

            # store each sublist of a batch in the results:
            # outputs2[start_original:end_original] = output_tensor.tolist()
            # targets2[start_original:end_original] = target.tolist()

            # store each sublist of a batch in the results:
            outputs2.extend(output_tensor.tolist())
            targets2.extend(target.tolist())

            # flatten the output:
            output_flattened = output_tensor.flatten()
            outputs[start_index:end_index] = output_flattened

            # flatten the targets:
            target_flattened = target.flatten()
            targets[start_index:end_index] = target_flattened

    # map to zeroes and ones:
    outputs2 = [[int(element >= 0.5) for element in sublist] for sublist in outputs2]
    # targets2 = [[int(element >= 0.5) for element in sublist] for sublist in targets2]

    if eval:
        o = np.array(outputs2).flatten()
        t = np.array(targets2).flatten()
        confusion_m(preds=o, targets=t)

    # convert to string, because DataFrame expects it otherwise dimensions don't fit!
    outputs2 = [str(sublist) for sublist in outputs2[:len(loader.dataset) * loader.batch_size]]
    targets2 = [str(sublist) for sublist in targets2[:len(loader.dataset) * loader.batch_size]]

    loss = criterion(outputs, targets.float())
    if approach == 'ast':
        loss = criterion(outputs, targets.half())
    val_loss.append(loss.item())

    targets = targets.numpy()
    outputs = (outputs >= 0.5).float().cpu().numpy()  # binary mapping

    results = {
        key: metrics[key](targets, outputs) for key in metrics.keys()
    }
    return results, targets2, outputs2, val_loss


def confusion_m(preds, targets):
    # Calculate confusion matrix
    cm = confusion_matrix(targets, preds)

    # Create heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

    # Add labels and title
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # Show plot
    plt.show()


def evaluate_class_balance(splitcsv='data/metadata/sed2/wdw200/combined/split.csv'):
    split = pd.read_csv(splitcsv)
    print(split.active.value_counts())


# ###################
# plots :
# ###################


def model_plot_reg(y_pred_path, config=None, save=False):
    """
    Plots predictions (red) vs. the actual stepcount (green) of fold_n

    Args:
        y_pred_path: path to the folder Epoch_last containing the test.csv file
        save: save to experiment folder if true
        config: info about hyperparameters

    """
    plot = pd.read_csv(y_pred_path)
    plt.plot(plot['steps'], 'g', label='truth', linewidth=0.2, linestyle='dashed')
    plt.plot(plot['predictions'], 'r', label='prediction', linewidth=0.2, linestyle='dashed')
    plt.suptitle(f"Predictions vs. Truth \n {config}")
    plt.xlabel('window')
    plt.ylabel('steps')
    plt.yscale('linear')  # 'log'
    plt.legend(loc=2, prop={'size': 6})
    if save:
        try:
            plt.savefig(y_pred_path.removesuffix(f'Epoch_{config.epochs}/test.csv') + 'pvt.png')
        except RuntimeError as e:
            print(e)
    plt.show()


def model_plot_loss(loss, save=False, path=None):
    """scatterplot, train loss"""
    lossframe = pd.DataFrame(loss, columns=['loss'])
    lossframe.plot(ylim=(0, 0.6), figsize=(16, 8), alpha=0.5, marker='.', grid=True, yticks=(0, 0.25, 0.5))
    if save and path is not None:
        try:
            plt.savefig(path + 'loss.png')
        except RuntimeError as e:
            print(e)


def model_plot_losses(train_loss, val_loss):
    """scatterplot, train and val loss"""
    d = {'train_loss': train_loss, 'val_loss': val_loss}
    lf = pd.DataFrame(d, columns=['train_loss', 'val_loss'])
    lf.plot(ylim=(0, 0.6), figsize=(16, 8), alpha=0.5, marker='.', grid=True, yticks=(0, 0.25, 0.5))


def plot_losses(train_loss, val_loss, epochs, savepath=None):
    """lineplot, train and val loss"""
    epochs = range(0, epochs)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if savepath is not None:
        try:
            plt.savefig(savepath)
        except RuntimeError as e:
            print(e)
    plt.show()


def regplot(pred_path, alpha=1.0, title=None, collage=False, descriptor='model'):

    if type(pred_path) == str:
        #print('simple regplot')
        preds = pd.read_csv(pred_path)
        # draw regplot
        sns.regplot(x="steps", y="predictions", data=preds)#, x_estimator=np.mean)
        # sns.jointplot(x="steps", y="predictions", data=preds, kind="reg", x_estimator=np.mean)
        if title:
            plt.title(title)

    if type(pred_path) == list and collage:
        #print('collaged regplot')
        assert len(pred_path) % 2 == 0, 'uneven subplots'
        half = len(pred_path) // 2
        fig, axs = plt.subplots(half, half)
        for i, pred in enumerate(pred_path):
            p = pd.read_csv(pred)
            axs[i, i] = sns.regplot(x="steps", y="predictions", data=p, x_estimator=np.mean)
            axs[i, i].set_title(title[i])

    if type(pred_path) == list:
        #print('combined regplot')
        assert type(title) == list, 'need list of titles to accompany each fit'  # another option: dictionary
        dataframes = [pd.read_csv(p) for p in pred_path]  # read all dataframes (test.csv) in list
        for i, frame in enumerate(dataframes):  # add unique descriptor to each
            frame[descriptor] = title[i]

        gt = dataframes[0].copy(deep=True)
        gt['predictions'] = gt['steps']
        gt[descriptor] = 'truth'
        dataframes.append(gt)
        total = pd.concat(dataframes)  # concat to one
        #sns.set_style("whitegrid", {'grid.linestyle': '--'})

        # sns.regplot(x="steps", y="predictions", data=total, hue="frame", x_estimator=np.mean)

        g = sns.FacetGrid(total, hue=descriptor, height=7)
        # sns.regplot(x="steps", y="predictions", data=gt, scatter_kws={'alpha':0.5}, x_estimator=np.mean)  # groundtruth
        g.map(sns.regplot, "steps", "predictions", data=total, x_estimator=np.mean)
        g.add_legend()
        plt.grid()
        # plt.title('All models, split 3')
        plt.show()


        # scatter_kws={'alpha':0.5}, x_estimator=np.mean
        # sns.implot(x="steps", y="predictions", data=total, hue="'model, s3'", x_estimator=np.mean)
        # plt.show()


        g = sns.FacetGrid(total, hue=descriptor, height=7)
        # sns.regplot(x="steps", y="predictions", data=gt, scatter_kws={'alpha':0.5})  # groundtruth
        g.map(sns.regplot, "steps", "predictions", data=total, x_estimator=np.mean)
        g.add_legend()
        plt.show()

        sns.set(rc={"figure.figsize": (10, 6)})
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        sns.lmplot(x="steps", y="predictions", hue=descriptor, data=total, x_estimator=np.mean, height=7)
        plt.show()


    #plt.show()


def plot_actual_vs_pred(pred_path, title=None):
    data = pd.read_csv(pred_path)
    y = data['steps']
    p = data['predictions']
    fig, ax = plt.subplots()
    ax.scatter(y, p, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    if title:
        plt.title(title)
    plt.show()


def plot_label_dist(d_path='data/metadata/wdw5s/fold0/fold0.csv'):
    data = pd.read_csv(d_path)
    y = data['steps']
    plt.figure(figsize=(6, 4))
    # sns.kdeplot(y, shade=True)
    sns.histplot(data=y, kde=True)
    # x-axis:
    plt.xlabel('steps')
    plt.xlim(1, 20)
    plt.title('Distribution of Labels')
    # display uneven ticks:
    xticks = [i for i in range(20) if i % 2 == 1]
    # y-axis:
    plt.xticks(xticks)
    plt.ylim(0, 1e5)
    plt.ylabel('windows')
    plt.yscale('symlog')
    #yticks = [1e0, 1e3, 1e5]
    #plt.yticks(yticks)
    # plot:
    #plt.figure(figsize=(6, 4), dpi=80)
    plt.savefig(fname='tools/analysis/img/dist.pdf', format='pdf')
    plt.show()


def plot_label_dist2(d_path='data/metadata/wdw5s/fold0/fold0.csv'):
    data = pd.read_csv(d_path)
    y = data['steps']

    plt.figure(figsize=(6, 4))
    # Plot the histogram of ground truth
    plt.hist(y, bins=60, alpha=0.5)

    # x-axis:
    plt.xlabel('steps (int)')
    plt.xlim(1, 20)
    plt.title('Distribution of Labels (s0)')
    # display uneven ticks:
    xticks = [i for i in range(20) if i % 2 == 1]
    # y-axis:
    plt.xticks(xticks)
    plt.ylim(0, 1e5)
    plt.ylabel('windows')
    plt.yscale('symlog')

    # Scale the y-axis with equally distributed spaces between y-ticks
    tick_positions = [10, 1e2, 1e3, 1e4, 1e5]
    #tick_labels = ['10', '1e2', '1e4']
    plt.yticks(tick_positions)#, tick_labels)

    # Add a legend
    plt.legend()
    plt.savefig(fname='tools/analysis/img/dist2.pdf', format='pdf')
    plt.show()


# ###################
# metrics :
# ###################


def mae_baseline():
    window_size = [5, 10, 20]
    for size in window_size:
        print(f'wdw-size {size} - mae')
        basepath = f'data/metadata/wdw{size}/'
        datapaths = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
        for path in datapaths:
            df_train = pd.read_csv(basepath + path + '/train.csv')
            df_test = pd.read_csv(basepath + path + '/test.csv')
            df_dev = pd.read_csv(basepath + path + '/devel.csv')
            print(path, 'train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))
            print(path, 'train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))


def mae_baseline_dynamic(datapath):
    df_train = pd.read_csv(osp.join(datapath, 'train.csv'))
    df_test = pd.read_csv(osp.join(datapath, 'test.csv'))
    return mae([df_train['steps'].mean()] * len(df_test), df_test['steps'])


def rmse(y_pred_path):
    a = pd.read_csv(y_pred_path)
    b = a['predictions']
    c = a['steps']
    return np.sqrt(torch.nn.MSELoss(b, c))


def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


def r2_score(y_pred_path):
    a = pd.read_csv(y_pred_path)
    b = a['predictions']
    c = a['steps']
    return sklearn.metrics.r2_score(b, c)


def percent_diff(base, new):
    diff = 100 * (base - new) / base
    return '{:.4f} %'.format(diff)


# ###################
# audio util (more in audio_features.py)
# ###################


def plot_audio(filepath=None, multiple=False):
    """
    Args:
        filepath: (optional) specify path to the .npy file which contains mel-spectogram data from repository root
    Returns: path to the spectogram which was displayed
    """
    spectogram_path = 'data/melspec_data/sed2/wdw200/split/'
    dirs = os.listdir(spectogram_path)

    if multiple:
        filepaths = dirs[:10]
        for f in filepaths:
            print("spectogram-name:", f)
            sp = spectogram_path + f
            s = np.load(sp)
            s = 20. * np.log10(np.abs(s) / 10e-6)  # amplitude to decibel
            # plt.figure(figsize=(15, 7.5))
            plt.imshow(np.transpose(s), origin="lower", aspect="auto", cmap='inferno', interpolation="none")
            plt.colorbar()
            plt.xlabel("time (s)")
            plt.ylabel("frequency (hz)")
            plt.show()
    else:
        if filepath is None:
            filepath = dirs[np.random.randint(1, len(dirs))]
            spectogram_path = spectogram_path + filepath
        else:
            spectogram_path = filepath
            filepath = dirs[np.random.randint(1, len(dirs))]
            spectogram_path = spectogram_path + filepath
        print("spectogram-name:", filepath)
        s = np.load(spectogram_path)
        s = 20. * np.log10(np.abs(s) / 10e-6)  # amplitude to decibel
        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(s), origin="lower", aspect="auto", cmap='inferno', interpolation="none")
        plt.colorbar()
        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.show()

    return spectogram_path


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect, cmap='inferno')
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_window(
        windowed_annotation_csv='data/metadata/sed2/wdw200/combined/split.csv',
        datadir='data\kirun-data-by-runner',
        count=5
):
    # windowed_annotation_csv='data/metadata/wdw5/fold0/fold0.csv'

    df = pd.read_csv(windowed_annotation_csv)

    sample_rate = 16000
    window_length = 320
    n_fft = window_length  # 320
    hop_length = 160
    f_min = 50  # 50
    f_max = 8000  # 8000
    n_mels = 64  # 64

    transform_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,  # 320
        f_min=f_min,  # 50
        f_max=f_max,  # 8000
        n_mels=n_mels,  # 64
    )

    transform_spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,  # 320
        win_length=window_length,
        hop_length=hop_length,
    )

    mel_energy = []
    combined = []

    df.set_index(['file', 'start', 'end', 'active'], inplace=True)
    for counter, (file, start, end, active) in enumerate(df.index):
        offset = start
        duration = end - start

        audio, fs = af.read(
            os.path.join(datadir, file),
            always_2d=True,
            offset=offset,
            duration=duration
        )
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdims=True)
        if fs != 16000:
            audio = torchaudio.transforms.Resample(fs, 16000)(torch.from_numpy(audio))
        else:
            audio = torch.from_numpy(audio)

        # USE A BANDPASS FILTER TO EXTRACT STEP SPIKES:
        # define the filter parameters
        # f_low = 300  # lower frequency of the passband
        # f_high = 600  # higher frequency of the passband
        # Wn = [2 * f_low / fs, 2 * f_high / fs]  # normalized cutoff frequencies
        # N = 4  # filter order
        # # design the filter
        # b, a = butter(N, Wn, btype='band')
        # # apply the filter
        # audio = filtfilt(b, a, audio)
        # audio = torch.from_numpy(audio.copy())
        # audio = audio.float()

        # print(audio.shape)
        # print(audio[:, :800].shape)
        # audio = audio[:, :800]
        # exit(0)

        mel_spec = transform_melspec(audio.float())
        #stretch = torchaudio.transforms.TimeStretch(hop_length=hop_length, n_freq=n_mels)
        #rate = 0.5
        #mel_spec = stretch(mel_spec, rate)

        # spec = transform_spec(audio)

        # mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        # combined.append(mel_spec)

        #mel_energy.append(torch.mean(mel_spec, dim=-1).numpy())

        plot_spectrogram(mel_spec.permute(1,2,0), title=f'active:{active}')
        #exit(0)
        if counter + 2 > count:
            break

    m = torch.cat(combined, dim=-1)
    print(m.shape)
    plot_spectrogram(m.permute(1, 2, 0), title=f'Concatenated along time axis')

    # Convert the Mel energy to a numpy array
    mel_energy = np.squeeze(mel_energy)

    # Create a plot of the Mel energy for each audio file
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(mel_energy.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Audio files')
    ax.set_ylabel('Mel frequency bands')
    plt.show()


def plot_waveform(filepath=None, sr=16000, title="Waveform", multiple=True):
    """
    Args:
        filepath: (optional) specify path to the .npy file which contains mel-spectogram data from repository root
    Returns: path to the spectogram which was displayed
    """
    waveform_path = 'data/waveform_data/sed2/wdw200/split/'
    dirs = os.listdir(waveform_path)

    if multiple:
        filepaths = dirs[:10]
        for f in filepaths:
            print("waveform-name:", f)
            wf = waveform_path + f
            waveform = np.load(wf)
            num_channels, num_frames = waveform.shape
            time_axis = torch.arange(0, num_frames) / 16000
            figure, axes = plt.subplots(num_channels, 1)
            axes.plot(time_axis, waveform[0], linewidth=1)
            axes.grid(True)
            figure.suptitle('Waveform')
            plt.show(block=False)
    else:
        if filepath is None:
            dirs = os.listdir(waveform_path)
            filepath = dirs[np.random.randint(1, len(dirs))]
            waveform_path = waveform_path + filepath
        else:
            waveform_path = filepath
        print("waveform-name:", filepath)
        waveform = np.load(waveform_path)
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / 16000
        figure, axes = plt.subplots(num_channels, 1)
        axes.plot(time_axis, waveform[0], linewidth=1)
        axes.grid(True)
        figure.suptitle('Waveform')
        plt.show(block=False)


def plot_wv(waveform, title=None):
    plt.plot(waveform)
    plt.xlabel('sample rate * time')
    plt.ylabel('energy')
    if title is not None:
        plt.title(title)
    plt.show()


def to_mono(wav, rand_ch=False):
    if wav.ndim > 1:
        if rand_ch:
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            wav = np.mean(wav, axis=-1)
    return wav


# ###################
# DNN util :
# ###################


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calc_conv_out(kernel: int | tuple, padding: int | tuple = (0, 0), stride: int | tuple = (1, 1),
                  dilation: int | tuple = (1, 1), h_in=500, w_in=64):
    """
    References: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    Input: (C_in, H_in, W_in)
    Output: (C_out, H_out, W_out)

    C_in, C_out given.

    Args:
        padding:
        kernel:
        stride:
        dilation:
        h_in:
        w_in:

    Returns:
        H_out * W_out
    """

    h_out = math.floor(((h_in + (2 * padding[0]) - dilation[0] * (kernel[0] - 1) - 1) / stride[0]) + 1)
    w_out = math.floor(((w_in + (2 * padding[1]) - dilation[1] * (kernel[1] - 1) - 1) / stride[1]) + 1)

    return h_out, w_out

def plot_gradient_descent_example():
    """

    Define a simple function f(x, y) = x^2 + y^2 to optimize using gradient descent.
    Define the partial derivatives of the function df_dx and df_dy.
    Define the gradient_descent function to perform the gradient descent algorithm and

    Return the history of the optimization process.
    Finally, we run the gradient descent algorithm on the function with a starting point of (-4, 4),
    a learning rate of 0.1, and 50 iterations, and plot the results using Matplotlib and the Axes3D module.

    Returns:

    """
    from mpl_toolkits.mplot3d import Axes3D

    # Define the function to optimize
    def f(x, y):
        return x ** 2 + y ** 2

    # Define the partial derivatives of the function
    def df_dx(x, y):
        return 2 * x

    def df_dy(x, y):
        return 2 * y

    # Define the gradient descent algorithm
    def gradient_descent(eta, num_iters, start=(-4,4)):
        x, y = start
        history = [(x, y, f(x, y))]
        for i in range(num_iters):
            x = x - eta * df_dx(x, y)
            y = y - eta * df_dy(x, y)
            history.append((x, y, f(x, y)))
        return history

    # Define the starting point and learning rate
    start = (-4, 4)
    alpha = 0.1
    num_iters = 50

    # Run gradient descent and extract the results
    history = gradient_descent(start, alpha, num_iters)
    x_vals, y_vals, z_vals = zip(*history)

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    ax.plot(x_vals, y_vals, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    plt.show()


# ###################
# ANALYSIS:

supported_metrics = ['MAE', 'CC', 'CCC', "MSE", 'ACC', 'train_loss', 'val_loss']

def read_log(log_path, epoch_count=None, metric='MAE'):
    """
    Results of trainings are stored in logfiles. This function reads a log_path. Can be iterated over.
    Args:
        epoch_count: number of metric-values
        log_path: path to logfile (.log)
        metric: currently works for MAE/MSE/ACC. CC and CCC are ambiguos .

    Returns: list of metric with size of epochs of that training

    """
    assert metric in supported_metrics
    overall = True if epoch_count is None else False

    metric_storage = []
    with open(log_path) as log:
        lines = log.readlines()

    i = 0
    for line in lines:
        if metric in line:
            if not overall and i == epoch_count:
                return metric_storage
            # distinguish between CC & CCC
            if metric == 'CC' and 'CCC:' not in line:
                result = re.search("\d+\.\d+", line)
                if result:
                    result = float(result.group())
                    metric_storage.append(result)
                i += 1
                # print(line)
            elif metric != 'CC':
                result = re.search("\d+\.\d+", line)
                result = float(result.group())
                metric_storage.append(result)
                i += 1

    return metric_storage


# 'Playground' contains:
# - data analysis
# - plots for sound, regression, sed, ...

def Playground():

    # ##########################
    # ANALYSE DATA :
    # ##########################

    analyse_d = False
    if analyse_d:

        plot_label_dist()
        plot_label_dist2()

        analyse_mae = False
        if analyse_mae:

            basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw5s\fold0'
            df_train = pd.read_csv(basepath + '/train.csv')
            df_dev = pd.read_csv(basepath + '/devel.csv')
            print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            df_test = pd.read_csv(basepath + '/test.csv')
            print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))

            basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw5s\fold1'
            df_train = pd.read_csv(basepath + '/train.csv')
            df_dev = pd.read_csv(basepath + '/devel.csv')
            # print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            df_test = pd.read_csv(basepath + '/test.csv')
            print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))

            basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw5s\fold2'
            df_train = pd.read_csv(basepath + '/train.csv')
            df_dev = pd.read_csv(basepath + '/devel.csv')
            # print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            df_test = pd.read_csv(basepath + '/test.csv')
            print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))

            basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw5s\fold3'
            df_train = pd.read_csv(basepath + '/train.csv')
            df_dev = pd.read_csv(basepath + '/devel.csv')
            # print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            df_test = pd.read_csv(basepath + '/test.csv')
            print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))

            basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw5s\fold4'
            df_train = pd.read_csv(basepath + '/train.csv')
            df_dev = pd.read_csv(basepath + '/devel.csv')
            # print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            df_test = pd.read_csv(basepath + '/test.csv')
            print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))

            # basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw10\fold0'
            # df_train = pd.read_csv(basepath + '/train.csv')
            # df_dev = pd.read_csv(basepath + '/devel.csv')
            # print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            # df_test = pd.read_csv(basepath + '/test.csv')
            # print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))
            #
            # basepath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw20\fold0'
            # df_train = pd.read_csv(basepath + '/train.csv')
            # df_dev = pd.read_csv(basepath + '/devel.csv')
            # print('train-dev', mae([df_train['steps'].mean()] * len(df_dev), df_dev['steps']))
            # df_test = pd.read_csv(basepath + '/test.csv')
            # print('train-test', mae([df_train['steps'].mean()] * len(df_test), df_test['steps']))


        analyse_d_depth = False
        if analyse_d_depth:
            # ALL RUNS:
            steppath = r'data/metadata/original/steps.csv'
            data = pd.read_csv(steppath)
            runs = data['file'].unique()
            print(len(runs))  # 197 annotated runs in steps.csv

            # RUNS CONTAINED IN SPLITS:
            splitsteppath = r'data/metadata/wdw5/fold0/fold0.csv'
            data = pd.read_csv(splitsteppath)
            print(len(data[
                          'file'].unique()))  # 188 annotated runs in fold0, fold1, fold2, ... This number is representative for our training.

            # RUNNERS:
            metap = 'data/metadata/original/meta.csv'
            meta = pd.read_csv(metap)
            # print(meta.head())

            root = 'data/kirun-data-by-runner/'
            dirs = os.listdir(root)
            runners = []
            for ses in dirs:
                path = osp.join(root, ses)
                runners.extend(os.listdir(path))
            runners = set(runners)  # destroy duplicates

            usedrunners = pd.DataFrame()
            for runner in runners:
                usedrunners = usedrunners.append(meta[meta['runner'].str.contains(runner)], ignore_index=True)

            # sort by age descending:22

            print(usedrunners.sort_values(['Age']))
            print(np.mean(usedrunners['Experience']), np.max(usedrunners['Experience']),
                  np.min(usedrunners['Experience']))  # 10.490196078431373, 53.0, 0.0
            print(np.mean(usedrunners['BMI']))  # 22.920272294955666
            print(np.mean(usedrunners['Weight']))  # 68.67450980392157
            print(np.mean(usedrunners['Age']))  # 39.31372549019608

            # check speaker independancy:
            for i in range(5):

                fold = f'data/metadata/original/fold{i}'
                train = fold + '/steps_train.csv'
                test = fold + '/steps_test.csv'
                val = fold + '/steps_dev.csv'

                train = pd.read_csv(train)
                test = pd.read_csv(test)
                val = pd.read_csv(val)

                train0_runner = set(train['runner'].unique())
                test0_runner = set(test['runner'].unique())
                val0_runner = set(val['runner'].unique())

                print('train-test:', train0_runner.intersection(test0_runner))  # set() == empty set == none lie in the intersection
                print('train-val:', train0_runner.intersection(val0_runner))  # set() == empty set == none lie in the intersection

            fold = f'data/metadata/wdw5/combined'
            train = fold + '/train.csv'
            test = fold + '/test.csv'
            val = fold + '/devel.csv'

            train = pd.read_csv(train)
            test = pd.read_csv(test)
            val = pd.read_csv(val)

            print(len(train), len(test), len(val))

            train0_runner = set(train['file'].unique())
            test0_runner = set(test['file'].unique())
            val0_runner = set(val['file'].unique())

            print('train-test:',
                  train0_runner.intersection(test0_runner))  # set() == empty set == none lie in the intersection
            print('train-val:',
                  train0_runner.intersection(val0_runner))  # set() == empty set == none lie in the intersection
            print('train-train (test):',
                  train0_runner.intersection(train0_runner))  # set() == empty set == none lie in the intersection

    # ##########################
    # ANALYSE REGRESSION :
    # ##########################

    analyse_r = True
    if analyse_r:

        analyse_log = False
        if analyse_log:
            #############
            # METRICvsEPOCH PLOT FOR ALL ARCHITECTURES (xyz.log)

            # all epoch 20:
            fnns = {
                'fnn': 'data/results/wdw5/dnn/fold0/mfcc_Adam_e20_b32/20230301_170529_logs/0000.log',
            }
            cnns = {
                'psla': 'data/results/wdw5_m128/psla/fold0/Adam_e50_b32_default/20230320_185412_logs/0000.log',
                'cnn6': 'data/results/wdw5/cnn6/fold0/Adam_e50_b32/20230307_150353_logs/0000.log',
                'cnn10': 'data/results/wdw5/cnn10/fold0/Adam_e100_b32_sa/20230303_022158_logs/0000.log',
                'cnn14': 'data/results/wdw5/cnn14/fold0/SGD_e50_b16_fa/20230227_030627_logs/0000.log',
                'wvmlmcnn14': 'data/results/wdw5/wvlmcnn14/fold0/SGD_e50_b32/20230306_020226_logs/0000.log',
                'mobilenetv1': 'data/results/wdw5/mobilenet/fold0/SGD_e20_b32/20230318_014930_logs/0000.log',
            }
            crnns = {
                'facrnn': 'data/results/wdw5/facrnn/fold0/Adam_e50_b32/20230302_020340_logs/0000.log',
            }
            rnns = {
                'lstm': 'data/results/wdw5/lstm/fold0/Adam_e50_b16/20230303_194007_logs/0000.log',
            }
            transformer = {
                'hts': 'data/results/wdw5/hts/fold0/AdamW_e50_b32_1e-4_filteraug/20230224_184951_logs/0000.log',
                'passt': 'data/results/wdw5/passt/fold0/AdamW_e50_b32/20230302_150929_logs/0000.log',
                'ast': 'data/results/wdw5/ast/fold0/Adam_e50_b32_chroma_stft/20230315_193049_logs/0000.log',
            }

            transformer_all = {

            }

            dicts = {
                **fnns, **cnns, **crnns, **rnns, **transformer
            }
            colordicts = {
                'red': cnns,
                'blue': transformer,
                'green': crnns,
                'orange': rnns,
                'purple': fnns,
            }

            all_model_logs = []

            def plot_model_dicts(_dicts, epoch_count, usecolordict=True, metric='MAE', **kwargs):
                """
                Args:
                    _dicts: key: modelname, value: path to logfile
                    epoch_count: number of epochs a metric should be tracked
                    usecolordict: for all architectures: color highlights the given architecture
                    metric: the metric the log contains and should be read from
                Returns:

                """
                # For each model we get a list with metrics plotted over epoch_count (learning behavior)
                for model in _dicts:
                    logs = read_log(log_path=_dicts[model], metric=metric, epoch_count=epoch_count)
                    color = None  # Initialize the color variable to None
                    if usecolordict:
                        # Selected color based on architecture
                        for c, models in colordicts.items():
                            if model in models:
                                color = c
                                break  # Exit the loop once a match is found
                    plt.plot(logs, color=color, label=model, **kwargs)
                    plt.xlabel('Epoch')
                    plt.ylabel(metric)
                    all_model_logs.append(logs)  # for multiple metrics

            # for saving the image:
            save = True
            savepath = 'data/img/plots/'
            # metric = 'train_loss'

            # ALL ARCHITECTURES:
            metrics = ['MAE', 'MSE', 'CC', 'train_loss']
            for m in metrics:
                plot_model_dicts(dicts, epoch_count=60, usecolordict=False, metric=m)
                plt.grid()
                if m in ['MAE', 'MSE']:
                    plt.gca().invert_yaxis()  # invert y-axis
                plt.title(f'{m} vs Epoch: Architectures in comparison')
                if m == 'MAE':
                    plt.legend(loc="lower left", fontsize=9)
                if m == 'train_loss':
                    plt.legend(loc="upper right", fontsize=9)
                if save:
                    plt.savefig(savepath)  # overwrites old for copying
                plt.show()

            metric = 'MAE'
            # ALL CNNS:
            plot_model_dicts(cnns, epoch_count=60, usecolordict=False, metric=metric)
            plt.grid()
            if metric in ['MAE', 'MSE']:
                plt.gca().invert_yaxis()  # invert y-axis
            plt.title(f'{metric} vs Epoch: CNNs')
            plt.legend(loc="lower left", fontsize=9)
            if save:
                plt.savefig(savepath)  # overwrites old for copying
            plt.show()

            # ALL TRANSFORMER:
            plot_model_dicts(transformer, epoch_count=None, usecolordict=False, metric=metric)
            plt.grid()
            if metric in ['MAE', 'MSE']:
                plt.gca().invert_yaxis()  # invert y-axis
            plt.title(f'{metric} vs Epoch: Transformer')
            plt.legend(loc="lower left", fontsize=9)
            if save:
                plt.savefig(savepath)  # overwrites old for copying
            plt.show()

            # psla_pretrain = {
            #     'psla_imagenet': 'data/results/wdw5_m128/psla/fold0/Adam_e20_b32_pretrain/20230309_122852_logs/0000.log',
            #     'psla_randomweights': 'data/results/wdw5_m128/psla/fold0/Adam_e50_b32/20230309_075212_logs/0000.log',
            #     'psla_audioset': 'data/results/wdw5_m128/psla/fold0/Adam_e20_b32_as_pretrain/20230317_223408_logs/0000.log',
            #     'psla_fsd50k': 'data/results/wdw5_m128/psla/fold0/Adam_e20_b32_fsd_pretrain/20230317_135453_logs/0000.log',
            # }
            #
            # metric = 'MAE'
            # plot_model_dicts(psla_pretrain, epoch_count=None, usecolordict=False, metric=metric)
            # plt.gca().invert_yaxis()  # invert y-axis
            # plt.title(f'{metric} vs Epoch: CNNs')
            # plt.legend(loc="lower right", fontsize=9)
            # if save:
            #     plt.savefig(savepath)  # overwrites old for copying
            # plt.show()

            cnn10_augment = {
                'no-augment': 'data/results/wdw5/cnn10/fold0/Adam_e50_b32_noaugment/20230320_144326_logs/0000.log',
                'filteraug': 'data/results/wdw5/cnn10/fold0/Adam_e50_b16_fa/20230226_223058_logs/0000.log',
                'specaugment': 'data/results/wdw5/cnn10/fold0/Adam_e100_b32_sa/20230303_022158_logs/0000.log',
                # 'mixup': 'data/results/wdw5/cnn10/fold0/Adam_e50_b32_mixup/20230320_174735_logs/0000.log'
                'mixup@0.1': 'data/results/wdw5/cnn10/fold0/Adam_e50_b32_mixup0.1/20230323_033250_logs/0000.log',
                'mixup@0.3': 'data/results/wdw5/cnn10/fold0/Adam_e50_b32_mixup0.3/20230323_033440_logs/0000.log',
                'mixup@0.7': 'data/results/wdw5/cnn10/fold0/Adam_e50_b32_mixup0.7/20230323_034125_logs/0000.log',
                'mixup@1.0': 'data/results/wdw5/cnn10/fold0/Adam_e50_b32_mixup1.0/20230323_031215_logs/0000.log',
            }

            metric = 'MAE'
            plot_model_dicts(cnn10_augment, epoch_count=None, usecolordict=False, metric=metric, linestyle='--')
            plt.grid()
            if metric in ['MAE', 'MSE']:
                plt.gca().invert_yaxis()  # invert y-axis
            plt.title(f'{metric} vs Epoch: CNN10 Augmentation')
            plt.legend(loc="lower left", fontsize=9)
            if save:
                plt.savefig(savepath)  # overwrites old for copying
            plt.show()

            metric = 'CC'
            plot_model_dicts(cnn10_augment, epoch_count=None, usecolordict=False, metric=metric, linestyle='--')
            plt.grid()
            if metric in ['MAE', 'MSE']:
                plt.gca().invert_yaxis()  # invert y-axis
            plt.title(f'{metric} vs Epoch: CNN10 Augmentation')
            plt.legend(loc="lower left", fontsize=9)
            if save:
                plt.savefig(savepath)  # overwrites old for copying
            plt.show()

            passt_all_splits = {
                'split0': 'data/results/wdw5/passt/fold0/AdamW_e50_b32/20230302_150929_logs/0000.log',
                'split1': 'data/results/wdw5/passt/fold1/AdamW_e50_b32_m128_w512/20230305_212009_logs/0000.log',
                'split2': 'data/results/wdw5/passt/fold2/AdamW_e20_b32/20230306_173103_logs/0000.log',
                'split3': 'data/results/wdw5/passt/fold3/AdamW_e20_b32.2/20230308_152405_logs/0000.log',
                'split4': 'data/results/wdw5/passt/fold4/AdamW_e50_b32/20230308_023928_logs/0000.log',
            }

            # metric = 'CC'
            # plot_model_dicts(passt_all_splits, epoch_count=None, usecolordict=False, metric=metric)
            # plt.grid()
            # if metric in ['MAE', 'MSE']:
            #     plt.gca().invert_yaxis()  # invert y-axis
            # plt.title(f'{metric} vs Epoch: Transformer')
            # plt.legend(loc="lower left", fontsize=9)
            # if save:
            #     plt.savefig(savepath)  # overwrites old for copying
            # plt.show()

        regplots = True
        if regplots:

            #############
            # 5 seconds, windowed at step
            #############
            # REGPLOT OF CROSS VAL ARCHS (test.csv) fold 0
            # passt_path = r'data/results/wdw5/passt/fold0/AdamW_e50_b32/Epoch_50/test.csv'
            # pred_path2 = r'data\results\wdw5\cnn10\fold0\Adam_e50_b16_fa\Epoch_50\test.csv'
            # pred_path3 = r'data/results/testing/psla/test.csv'
            # pred_path4 = 'data/results/wdw5/hts/fold0/AdamW_e50_b32_1e-4_filteraug/Epoch_50/test.csv'
            # pred_path5 = 'data/results/wdw5/wvlmcnn14/fold0/Adam_e100_b32/Epoch_100/test.csv'
            #
            # # REGPLOT OF ALL PASST (test.csv)
            # passt_f0 = 'data/results/wdw5/passt/fold0/AdamW_e50_b32/Epoch_50/test.csv'
            # passt_f1 = 'data/results/wdw5/passt/fold1/AdamW_e50_b32_m128_w512/test.csv'
            # passt_f2 = 'data/results/wdw5/passt/fold2/AdamW_e20_b32/Epoch_20/test.csv'
            # passt_f3 = 'data/results/wdw5/passt/fold3/AdamW_e20_b32.2/Epoch_20/test.csv'
            # passt_f4 = 'data/results/wdw5/passt/fold4/AdamW_e50_b32/Epoch_50/test.csv'
            # passt_paths = [passt_f0, passt_f1, passt_f2, passt_f3, passt_f4]
            # passt_names = ['PaSST-S, s0', 'PaSST-S, s1', 'PaSST-S, s2', 'PaSST-S, s3', 'PaSST-S, s4']
            #
            # # REGPLOT OF ALL PSLA (test.csv)
            # psla_f0 = 'data/results/wdw5_m128/psla/fold0/Adam_e50_b32_default/Epoch_50/test.csv'
            # psla_f1 = 'data/results/wdw5_m128/psla/fold1/Adam_e100_b32/Epoch_100/test.csv'
            # psla_f2 = 'data/results/testing/psla/test.csv'
            # psla_f3 = 'data/results/wdw5_m128/psla/fold3/Adam_e30_b32/Epoch_30/test.csv'
            # psla_f4 = 'data/results/wdw5_m128/psla/fold4/Adam_e30_b32/Epoch_30/test.csv'
            # psla_paths = [psla_f0, psla_f1, psla_f2, psla_f3, psla_f4]
            # psla_names = ['PSLA, s0', 'PSLA, s1', 'PSLA, s2', 'PSLA, s3', 'PSLA, s4']
            #
            # # REGPLOT OF ALL CNN10 (test.csv)
            # cnn10_f0 = 'data/results/wdw5_m128/psla/fold0/Adam_e50_b32_default/Epoch_50/test.csv'
            # cnn10_f1 = 'data/results/wdw5_m128/psla/fold1/Adam_e100_b32/Epoch_100/test.csv'
            # cnn10_f2 = 'data/results/testing/psla/test.csv'
            # cnn10_f3 = 'data/results/wdw5_m128/psla/fold3/Adam_e30_b32/Epoch_30/test.csv'
            # cnn10_f4 = 'data/results/wdw5_m128/psla/fold4/Adam_e30_b32/Epoch_30/test.csv'
            # cnn10_paths = [cnn10_f0, cnn10_f1, cnn10_f2, cnn10_f3, cnn10_f4]
            # cnn10_names = ['CNN10, s0', 'CNN10, s1', 'CNN10, s2', 'CNN10, s3', 'CNN10, s4']
            #
            # hts_f3 = 'data/results/testing/hts/fold3/test.csv'
            #
            # xticks = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
            # fold0 = [passt_path, pred_path2, pred_path3, pred_path4, pred_path5]
            # models = ['passt', 'cnn10', 'psla', 'hts-at', 'wvlmcnn14']
            #
            # fold3 = [passt_f3, cnn10_f3, psla_f3, hts_f3]
            #
            # regplot(fold0, title=models, descriptor='model, s0')
            # # regplot(fold3, title=models, descriptor='model, s3')
            # # exit(0)
            #
            # cnn10_sed_1s = 'data/results/sedconcat/sed_segmented1s/cnn10/fold0/Adam_e50_b32/Epoch_100/test.csv'
            # data = pd.read_csv(cnn10_sed_1s)
            # preds = data['predictions']
            # targ = data['activities']
            #
            # preds = [[element for element in sublist] for sublist in preds]
            # targ = [[element for element in sublist] for sublist in targ]
            #
            # confusion_m(preds, targ)

            # dataframes = [pd.read_csv(p) for p in fold3]  # read all dataframes (test.csv) in list
            # for i, frame in enumerate(dataframes):  # add unique descriptor to each
            #     frame['model'] = models[i]
            #
            # gt = dataframes[0].copy(deep=True)
            # gt['predictions'] = gt['steps']
            # gt['model, s0'] = 'truth'
            # dataframes.append(gt)
            # total = pd.concat(dataframes)
            #
            # sns.set(rc={"figure.figsize": (10, 6)})
            # sns.set_style("whitegrid", {'grid.linestyle': '--'})
            # sns.lmplot(x="steps", y="predictions", hue="model", data=total, x_estimator=np.mean, height=7)
            # plt.show()

            # passt = False
            # if passt:
            #
            #     preds = []
            #     targets = []
            #
            #     for path in passt_paths:
            #         dataframes = pd.read_csv(path)
            #         for index, row in dataframes.iterrows():
            #             preds.append(row['predictions'])
            #             targets.append(row['steps'])
            #
            #     dataframes = [pd.read_csv(p) for p in passt_paths]  # read all dataframes (test.csv) in list
            #     for i, frame in enumerate(dataframes):  # add unique descriptor to each
            #         frame['model'] = passt_names[i]
            #
            #     gt = dataframes[0].copy(deep=True)
            #     gt['predictions'] = gt['steps']
            #     gt['model'] = 'truth'
            #     dataframes.append(gt)
            #     total = pd.concat(dataframes)
            #
            #     # create the stripplot and set the xticks parameter to the list you created
            #     sns.set(rc={"figure.figsize": (10, 6)})
            #     sns.stripplot(data=gt, x="steps", y="steps", jitter=0.1, size=1, color='red')
            #     sns.stripplot(data=total, x="steps", y="steps", jitter=0.1, size=1)
            #     plt.title('PaSST-S, all 5 Splits')
            #     locs, labels = plt.xticks()
            #     plt.xticks(np.arange(0, max(gt['predictions']), step=2.0))
            #     plt.show()
            #
            #     exit()
            #
            #     # sns.set(rc={"figure.figsize": (10, 6)})
            #     # sns.set_style("whitegrid", {'grid.linestyle': '--'})
            #     # sns.lmplot(x="steps", y="predictions", hue="model", data=total, markers=['s', 'x', 'p', '+', '.'],
            #     #            x_estimator=np.mean, scatter_kws={'alpha':0.5}, height=7)
            #     # plt.title('PaSST-S, all 5 Splits')
            #     # plt.show()
            #
            #     # sns.regplot(x="steps", y="predictions", data=total, hue="frame", x_estimator=np.mean)
            #     g = sns.FacetGrid(total, hue="model", height=7)
            #     # sns.regplot(x="steps", y="predictions", data=gt, scatter_kws={'alpha':0.5}, x_estimator=np.mean)  # groundtruth
            #     g.map(sns.regplot, "steps", "predictions", data=total, x_estimator=np.mean)
            #     g.add_legend()
            #     plt.grid()
            #     plt.title('PaSST-S')
            #     plt.show()
            #
            #
            #
            #     sns.boxplot(data=total, x="steps", y="predictions")
            #     plt.title('PaSST-S, all 5 Splits')
            #     plt.show()
            #
            #     # scatter_kws={'alpha':0.5}, x_estimator=np.mean
            #     # sns.regplot(x="steps", y="predictions", data=total, hue="frame", x_estimator=np.mean)
            #     g = sns.FacetGrid(total, hue="model,", height=7)
            #     # sns.regplot(x="steps", y="predictions", data=gt, scatter_kws={'alpha':0.5})  # groundtruth
            #     g.map(sns.regplot, "steps", "predictions", data=total, x_estimator=np.mean)
            #     g.add_legend()
            #     # plt.grid()
            #     plt.show()
            #
            # psla = False
            # if psla:
            #
            #     preds = []
            #     targets = []
            #
            #     for path in psla_paths:
            #         dataframes = pd.read_csv(path)
            #         for index, row in dataframes.iterrows():
            #             preds.append(row['predictions'])
            #             targets.append(row['steps'])
            #
            #
            #     dataframes = [pd.read_csv(p) for p in psla_paths]  # read all dataframes (test.csv) in list
            #     for i, frame in enumerate(dataframes):  # add unique descriptor to each
            #         frame['model'] = psla_names[i]
            #
            #     gt = dataframes[0].copy(deep=True)
            #     gt['predictions'] = gt['steps']
            #     gt['model, s0'] = 'truth'
            #     # dataframes.append(gt)
            #     total = pd.concat(dataframes)
            #
            #     sns.set(rc={"figure.figsize": (10, 6)})
            #     sns.stripplot(data=gt, x="steps", y="predictions", jitter=0.2, size=1, color='red')
            #     sns.stripplot(data=total, x="steps", y="predictions", jitter=0.2, size=1)
            #     plt.title('PSLA-S, all 5 Splits')
            #     # plt.xticks(xticks)
            #     plt.show()
            #
            #     sns.set(rc={"figure.figsize": (10, 6)})
            #     sns.set_style("whitegrid", {'grid.linestyle': '--'})
            #     sns.lmplot(x="steps", y="predictions", hue="model", data=total, markers=['s', 'x', 'p', '+', '.'],
            #                x_estimator=np.mean, scatter_kws={'alpha':0.5}, height=7)
            #     plt.title('PSLA, all 5 splits')
            #     plt.show()
            #
            #     sns.boxplot(data=total, x="steps", y="predictions")
            #     plt.title('PSLA, all 5 splits')
            #     plt.show()
            #
            #
            #
            #     # scatter_kws={'alpha':0.5}, x_estimator=np.mean
            #     # sns.regplot(x="steps", y="predictions", data=total, hue="frame", x_estimator=np.mean)
            #     g = sns.FacetGrid(total, hue="model", height=7)
            #     # sns.regplot(x="steps", y="predictions", data=gt, scatter_kws={'alpha':0.5})  # groundtruth
            #     g.map(sns.regplot, "steps", "predictions", data=total, x_estimator=np.mean)
            #     g.add_legend()
            #     plt.show()
            #
            # cnn10 = False
            # if cnn10:
            #
            #     preds = []
            #     targets = []
            #
            #     for path in cnn10_paths:
            #         dataframes = pd.read_csv(path)
            #         for index, row in dataframes.iterrows():
            #             preds.append(row['predictions'])
            #             targets.append(row['steps'])
            #
            #     dataframes = [pd.read_csv(p) for p in cnn10_paths]  # read all dataframes (test.csv) in list
            #     for i, frame in enumerate(dataframes):  # add unique descriptor to each
            #         frame['model'] = cnn10_names[i]
            #
            #     gt = dataframes[0].copy(deep=True)
            #     gt['predictions'] = gt['steps']
            #     gt['model, s0'] = 'truth'
            #     # dataframes.append(gt)
            #     total = pd.concat(dataframes)
            #
            #     # create the stripplot and set the xticks parameter to the list you created
            #     sns.set(rc={"figure.figsize": (10, 6)})
            #     sns.stripplot(data=gt, x="steps", y="predictions", jitter=0.2, size=1, color='red')
            #     sns.stripplot(data=total, x="steps", y="predictions", jitter=0.2, size=1)
            #     plt.title('PaSST-S, all 5 Splits')
            #     plt.xticks(xticks)
            #     plt.show()
            #
            #     sns.stripplot(data=total, x="steps", y="predictions", jitter=0.2, size=1)
            #     plt.title('PSLA on all 5 splits')
            #     plt.show()
            #
            #     sns.set_style("whitegrid", {'grid.linestyle': '--'})
            #     sns.lmplot(x="steps", y="predictions", hue="model", data=total, markers=['s', 'x', 'p', '+', '.'],
            #                x_estimator=np.mean, scatter_kws={'alpha': 0.5})
            #     # plt.show()
            #     # scatter_kws={'alpha':0.5}, x_estimator=np.mean
            #     # sns.regplot(x="steps", y="predictions", data=total, hue="frame", x_estimator=np.mean)
            #     g = sns.FacetGrid(total, hue="model", height=7)
            #     # sns.regplot(x="steps", y="predictions", data=gt, scatter_kws={'alpha':0.5})  # groundtruth
            #     g.map(sns.regplot, "steps", "predictions", data=total, x_estimator=np.mean)
            #     g.add_legend()
            #     plt.show()

            #######################################
            # 5 seconds, windowed at time
            #######################################

            plot_label_dist2()

            # REGPLOT OF CROSS VAL ARCHS (test.csv) fold 0
            passt_path = 'data/results/wdw5s/passt/fold0/AdamW_e100_b32/Epoch_100/test.csv'
            wvlmcnn14_path = 'data/results/wdw5s/wvlmcnn14/fold0/Adam_e100_b32_asckpt_BEST/Epoch_100/test.csv'
            psla_path = 'data/results/wdw5s_m128/psla/fold0/Adam_e100_b32_ImNet2/Epoch_50/test.csv'
            htsat_path = 'data/results/wdw5s/hts/fold0/AdamW_e100_b32/Epoch_40/test.csv'
            fnn_path = 'data/results/wdw5s/dnn/fold0/Adam_e100_b32_mfcc,7/Epoch_100/test.csv'
            fold0 = [passt_path, wvlmcnn14_path, psla_path, htsat_path, fnn_path]
            models = ['passt', 'wvlmcnn14', 'psla', 'htsat', 'Fnn']

            dataframes = [pd.read_csv(p) for p in fold0]  # read all dataframes (test.csv) in list
            for i, frame in enumerate(dataframes):  # add unique descriptor to each
                frame['model'] = models[i]
            # create a groundtruth dataframe:
            gt = dataframes[0].copy(deep=True)
            # Set up the figure and axes
            fig, ax = plt.subplots(figsize=(6, 4))
            # Plot the histogram of ground truth
            ax.hist(gt['steps'], bins=60, alpha=0.5, label="labels (int)", color=None)
            # Plot the histograms of predictions
            for i, pred in enumerate(dataframes):
                ax.hist(pred['predictions'], bins=60, alpha=0.5, label=pred['model'])
            # Add labels and title
            ax.set_xlabel("steps")
            ax.set_ylabel("windows")
            ax.set_yscale('log')
            plt.xlim(0, 20)
            # display uneven ticks:
            xticks = [i for i in range(20) if i % 2 == 1]
            # y-axis:
            plt.xticks(xticks)
            plt.title("Distribution of Predictions (s0)")
            # Add a legend
            plt.legend()
            plt.savefig(fname='tools/analysis/img/dist3.pdf', format='pdf')
            plt.show()


    # ##########################
    # ANALYSE SED :
    # ##########################

    analyse_sed = False
    if analyse_sed:
        cnn10_sed_1s = 'data/results/sedconcat/sed_segmented1s/cnn10/fold0/Adam_e50_b32/Epoch_100/test.csv'
        data = pd.read_csv(cnn10_sed_1s)
        preds = data['predictions']
        targ = data['activities']
        # print(preds.head())
        preds = [element for sublist in preds for element in eval(sublist)]
        print(preds[:20])
        targ = [element for sublist in targ for element in eval(sublist)]
        print(targ[:20])
        confusion_m(preds, targ)

    # ##########################
    # PLOT SPECTROGRAMS :
    # ##########################

    # plot_spectogram(multiple=True)
    # plot_spectogram('data/melspec_data/wdw5/fold0/')
    # plot_waveform()

    # ##########################
    # ANALYSE BINARY CLASSIFICATION :
    # ##########################

    # plot_window()
    # evaluate_class_balance()

    # 'data/metadata/sed2/wdw200/combined/split.csv'
    # True     106120
    # False     95978
    # 'data/metadata/sed2/wdw50/combined/split.csv'
    # False 809823
    # True 115271

    # df = pd.read_csv('data/metadata/sed2/wdw200/combined/split.csv')
    # class_counts = df['active'].value_counts()
    # print('sed', class_counts)
    #
    # df = pd.read_csv('data/metadata/sed_concat/fold0/fold0.csv')
    # class_counts = df['activities'].apply(lambda x: pd.Series(list(x))).stack().value_counts()
    # print('sed_concat', class_counts)
    #
    # df = pd.read_csv('data/metadata/sed_concat20ms/fold0/fold0.csv')
    # class_counts = df['activities'].apply(lambda x: pd.Series(list(x))).stack().value_counts()
    # print('sed_concat_atomic', class_counts)

    # ##########################
    # TEST FUNCTIONS :
    # ##########################

    # p = sns.load_dataset("penguins")
    # print(p.head())
    #
    # # Create a dataset with two groups
    # p = p[p["sex"].isin(["Male", "Female"])]
    #
    # # Plot multiple regression fits for each group
    # g = sns.FacetGrid(p, hue="species", height=5)
    # g.map(sns.regplot, "bill_length_mm", "body_mass_g", x_estimator=np.mean)
    # g.add_legend()
    # plt.show()

    # PLOT CC EXAMPLES:
    # Function 1 with strong positive correlation
    # x1 = np.arange(0, 10, 0.1)
    # y1 = x1 + np.random.normal(0, 1, len(x1))
    # corr1 = np.corrcoef(x1, y1)[0, 1]
    # # Function 2 with moderate positive correlation
    # x2 = np.arange(0, 10, 0.1)
    # y2 = x2 + np.random.normal(0, 2, len(x2))
    # corr2 = np.corrcoef(x2, y2)[0, 1]
    # # Function 3 with weak positive correlation
    # x3 = np.arange(0, 10, 0.1)
    # y3 = x3 + np.random.normal(0, 4, len(x3))
    # corr3 = np.corrcoef(x3, y3)[0, 1]
    # # Plot the functions
    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # axs[0,0].scatter(x1, y1)
    # axs[0,0].set_title(f"Strong Correlation (r = {corr1:.2f})")
    # axs[0,1].scatter(x2, y2)
    # axs[0,1].set_title(f"Moderate Correlation (r = {corr2:.2f})")
    # axs[0,2].scatter(x3, y3)
    # axs[0,2].set_title(f"Weak Correlation (r = {corr3:.2f})")
    # axs[1,0].scatter(x1, -y1)
    # axs[1,0].set_title(f"Strong Correlation (r = {-corr1:.2f})")
    # axs[1,1].scatter(x2, -y2)
    # axs[1,1].set_title(f"Moderate Correlation (r = {-corr2:.2f})")
    # axs[1,2].scatter(x3, -y3)
    # axs[1,2].set_title(f"Weak Correlation (r = {-corr3:.2f})")
    # plt.show()

    # PLOT ACT:
    # act = torch.nn.ReLU()
    # x = torch.linspace(-10, 10, 100)
    # y = act(x).numpy()
    # # Plot the ReLU function
    # plt.plot(x.numpy(), y)
    # plt.xlabel('x')
    # plt.ylabel('ReLU(x)')
    # plt.title('ReLU Activation Function')
    # plt.show()

    # preds = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # targets = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #
    # conf_mat = confusion_matrix(targets, preds)
    # print(conf_mat)
    #
    # confusion_m(preds, targets)
    #
    # preds_s = np.array([preds for i in range(2)])
    # print(preds_s)
    # preds_s = preds_s.flatten()
    # print(preds_s)
    # targets_s = np.array([targets for i in range(2)]).flatten()
    #
    # #print(preds_s)
    # #print(np.transpose(preds_s))
    #
    #
    # # Transpose the arrays to get a list of predictions for each label
    # preds_t = np.sum(preds_s, axis=0)
    # targets_t = np.sum(targets_s, axis=0)
    #
    #
    # print(preds_t)
    #
    # conf_mat = confusion_matrix(targets_s, preds_s)
    # print(conf_mat)
    #
    # confusion_m(preds_s, targets_s)

    # plot_gradient_descent_example()

    # from scipy.stats import beta
    #
    # # Define the range of alpha values to plot
    # alphas = np.linspace(0.0, 1.0, num=11)
    #
    # # Set the plot style
    # plt.style.use('seaborn-darkgrid')
    #
    # # Generate the plot
    # fig, ax = plt.subplots(figsize=(8, 5))
    # for alpha in alphas:
    #     x = np.linspace(0, 1, num=100)
    #     y = beta.pdf(x, alpha, alpha)
    #     ax.plot(x, y, label=f'alpha = {alpha:.1f}')
    #
    # # Set the plot title and labels
    # ax.set_title('Beta Distribution for Different Alpha Values')
    # ax.set_xlabel('x')
    # ax.set_ylabel('PDF')
    # ax.legend()
    #
    # # Show the plot
    # plt.show()

# ###################

def main():
    pass

if __name__ == '__main__':
    Playground()
    pass
