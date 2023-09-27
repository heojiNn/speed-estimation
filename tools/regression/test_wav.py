import argparse
import math

import audobject
import audtorch
import numpy as np
import os
import pandas as pd
import random
import torch
import tqdm
import yaml
import time
import logging

from hear21passt.models.preprocess import AugmentMelSTFT
from hear21passt.base import get_basic_model, get_model_passt

from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import matplotlib.pyplot as plt

from tools.utils import (
    evaluate_regression, transfer_features,
    MinMaxScaler, percent_diff, mae_baseline_dynamic,
)

from tools.datasets import (
    WavDataset, SpecDataset,
)

import models.ffnn as dnn
import models.cnn as cnn
import models.rnn as rnn
import models.crnn as crnn
import models.transformer as tfr





def parse_args():
    """Returns: cl args"""
    parser = argparse.ArgumentParser('KIRun Speed Training')
    parser.add_argument('--data-root', help='Path to (windowed) data annotation', required=True)
    parser.add_argument('--features', help='features.csv file', required=True)
    parser.add_argument('--approach', help='Test on untrained model')
    parser.add_argument('--checkpoint', help='Test on trained checkpoint, without it untrained model is used')
    parser.add_argument('--feat-type', default='wav',
                        choices=['chroma_stft', 'chroma_cqt', 'chroma_cens', 'mfcc',
                                 'spectral_contrast', 'poly_features', 'tonnetz', 'wav'])
    parser.add_argument('--savedir', help='path to directory to save results')
    args = parser.parse_args()
    return args


def init_model(_model, _device, _input_shape, sample_rate=16000, wdw_size=5):
    """
    Args:
        _input_shape:
        _model: Lowercase string, matching the name of the model
        _device: cpu or gpu ?
    Returns: Initialized Model
    """
    m = None

    if _input_shape is None:  # e.g. no spectral features -> default shape: (SR, WDW), i.e. (16000, 5)
        _input_shape = sample_rate, wdw_size

    if _model == '' or _model == 'dnn':
        m = dnn.DNN(
            input_size=_input_shape[0] * _input_shape[1],
            hidden_size=256,
            # n_layers=5,
            # dropout_p=0.2,
            batch_norm=False,
            augment=False
        )
    elif _model == 'cdnn':
        m = dnn.CDNN(output_dim=1, sigmoid_output=True)

    # -----------------------------------------------------------------------------------------------------------------#
    # CNNs
    elif _model == 'wvcnn14':
        m = cnn.WavegramCnn14(
            sample_rate=16000,
            window_size=500,
            hop_size=250,
            mel_bins=64,
            fmin=50,
            fmax=8192,
            classes_num=1
        )
    elif _model == 'wvlmcnn14':
        m = cnn.WavegramLogmelCnn14(
            sample_rate=16000,
            window_size=500,
            mel_bins=64,
            fmin=0,
            fmax=8192,
            classes_num=1
        )
    elif _model == 'wvlmcnn14lstm':
        m = crnn.WavegramLogmelCnn14LSTM(
            sample_rate=16000,
            window_size=500,
            hop_size=250,
            mel_bins=64,
            fmin=50,
            fmax=1024,
            classes_num=1,
            num_rnn_layers=1,
            rnn_hidden_dim=256,
            bidirectional=False,
            sigmoid_output = False,
            segmentwise = False,
        )

    # -----------------------------------------------------------------------------------------------------------------#
    # RNNs
    elif _model == 'gru':
        m = rnn.GRU(
            input_size=_input_shape[0] * _input_shape[1],
            num_layers=5,
            device=_device,
            hidden_size=256,
        )
    elif _model == 'lstm':
        m = rnn.LSTM(
            input_size=_input_shape[0] * _input_shape[1],
            num_layers=5,
            device=_device,
            hidden_size=256,
            batch_norm=False,
            dropout_p=0.2
        )

    # -----------------------------------------------------------------------------------------------------------------#
    # CRNNS


    # -----------------------------------------------------------------------------------------------------------------#
    # Tformers
    elif _model == 'passt':
        # Experimental Setup from 'Efficient Training of Audio Transformers with Patchout':
        # quote form paper:
        # We use the AdamW optimizer with weight decay of 10e-4, with a
        # maximum learning rate of 10e-5. We use a linear learning rate
        # decay from epoch 50 to 100, dropping the learning rate to 10e-7
        # and fine-tune the model for a further 20 epochs.
        # TODO: find out what stft settings are optimal
        mel = AugmentMelSTFT(
            n_mels=128,  # 128
            sr=16000,
            win_length=800,  # 800
            hopsize=320,
            n_fft=1024,  # 1024
            freqm=48,
            timem=192,
            htk=False,
            fmin=50,
            fmax=None,  # None
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000
        )
        net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=1)



        m = tfr.PasstBasicWrapper(mel=mel, net=net, mode="logits")
    elif _model == 'hts':
        import models.config_hts as config  # the input args can be defined here..
        # config args that are being used in the model: just for mel-spec extraction.
        # TODO: either export config to code or export code into one combined config applicable for all models...
        m = tfr.HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=1,
            window_size=config.htsat_window_size,
            config=config,
            depths=config.htsat_depth,
            embed_dim=config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head,
            use_checkpoint=False
        )

    return m


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
    args = parse_args()

    device = get_default_device()
    window_size_seconds = 5  # seconds
    window_size = window_size_seconds * 100
    sample_rate = 16000

    # Read data:
    df_train = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    df_train.set_index(['file', 'start', 'end'], inplace=True)
    df_test = pd.read_csv(os.path.join(args.data_root, 'test.csv'))
    df_test.set_index(['file', 'start', 'end'], inplace=True)

    # Scale steps
    scaler = MinMaxScaler(
        minimum=df_train['steps'].min(),
        maximum=df_train['steps'].max()
    )
    df_train['steps'] = df_train['steps'].apply(scaler.encode)
    df_test['steps'] = df_test['steps'].apply(scaler.encode)

    # Read data:
    features = pd.read_csv(args.features).set_index(['file', 'start', 'end'])
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), x)
    )

    # ------------------------------------------------------------------------------------------------------------------
    # DataLoaders :
    feature_type = args.feat_type
    input_shape = None

    # using default waveform (to feed into model(s), which can possibly extract further features..)
    # i.e. wvlmcnn14, which uses raw waveform and mel-specs in one forward pass..
    if feature_type == 'wav':
        logging.info('waveform input')
        db_args = {
            'features': features,
            'target_column': 'steps',
            'transform': audtorch.transforms.RandomCrop(sample_rate * window_size_seconds, axis=-1)
        }
        db_class = WavDataset

    # using spectral features to feed into model(s)
    else:
        n_fft = 2048
        hop_length = 512
        n_mfcc = 7
        n_chroma = 12
        signal_length = sample_rate * window_size_seconds
        tdim = math.floor((signal_length - n_fft) / hop_length) + 1

        if feature_type == 'chroma_stft':  # todo
            input_shape = (n_chroma, tdim)
        elif feature_type == 'chroma_cqt':  # todo
            input_shape = (n_chroma, tdim + 4)
        elif feature_type == 'chroma_cens':  # todo
            input_shape = (n_chroma, tdim)
        elif feature_type == 'mfcc':
            input_shape = (n_mfcc, tdim)  # (1, n_mfcc, tdim)
        elif feature_type == 'spectral_contrast':  # todo
            input_shape = (0, tdim)
        elif feature_type == 'poly_features':  # todo
            input_shape = (0, tdim)
        elif feature_type == 'tonnetz':  # todo
            input_shape = (n_chroma, tdim)
        else:
            print(f'Type {feature_type} not defined.')
            exit(-1)

        print(f'input_shape:{input_shape}')

        db_args = {
            'features': features,
            'target_column': 'steps',
            '_type': feature_type,
            'transform': audtorch.transforms.RandomCrop(sample_rate * window_size_seconds, axis=-1),
            'window_size': window_size_seconds,
            'sample_rate': sample_rate,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mfcc': n_mfcc,
            'n_chroma': n_chroma,
        }
        db_class = SpecDataset

    train_dataset = db_class(
        df_train.copy(),
        **db_args
    )
    x, y = train_dataset[0]

    test_dataset = db_class(
        df_test.copy(),
        **db_args
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )  # sample dimensions: ([1, 1, 80000])

    # ------------------------------------------------------------------------------------------------------------------

    # Model :
    model = init_model(args.approach, device, input_shape, sample_rate=sample_rate)
    if model is None:
        logging.info('Error initializing model! (please check args.approach) Exiting...')
        exit(-1)
    if args.checkpoint is not None:
        logging.info('Using Checkpoint..')
        initial_state = torch.load(args.checkpoint)
        model.load_state_dict(
            initial_state,
            strict=False
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Test:

    print('='*15, 'EVALUATION', '='*15)
    print(f'on {args.data_root}')
    print(f'MAE between train and test: ', mae_baseline_dynamic(args.data_root))

    # Evaluate Checkpoint:
    test_results, targets, outputs, _ = evaluate_regression(
        model,
        device,
        test_loader,
        transfer_features,
        scaler,
        args.approach
    )
    print(f'Best test results:\n{yaml.dump(test_results)}')
    print(percent_diff(base=mae_baseline_dynamic(args.data_root), new=test_results['MAE']))

    # SAVE TEST.CSV HERE:
    df_test['predictions'] = outputs
    df_test['steps'] = targets
    df_test.reset_index().to_csv(os.path.join(args.savedir, 'test.csv'), index=False)
    with open(os.path.join(args.savedir, 'test.yaml'), 'w') as fp:
        yaml.dump(test_results, fp)

    plt.plot(targets, 'g', label='truth', linewidth=0.2, linestyle='dashed')
    plt.plot(outputs, 'r', label='prediction', linewidth=0.2, linestyle='dashed')
    plt.yscale('linear')  # 'log'
    plt.legend(loc=2, prop={'size': 6})
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(targets, outputs, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(os.path.join(args.savedir, 'regplot.png'), dpi=150)
    plt.show()
































