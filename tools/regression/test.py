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


from tools.datasets import (
    CachedDataset,
)

from tools.utils import MinMaxScaler, evaluate_regression, transfer_features, percent_diff, mae_baseline_dynamic


def parse_args():
    """Returns: cl args"""
    parser = argparse.ArgumentParser('KIRun Speed Test')
    parser.add_argument('--data-root', help='Path to (windowed) data annotation', required=True)
    parser.add_argument('--features', help='features.csv file', required=True)
    parser.add_argument('--approach', help='Test on untrained model')
    parser.add_argument('--checkpoint', help='Test on trained checkpoint, without it untrained model is used')
    parser.add_argument('--nmels', default=64)
    parser.add_argument('--savedir', help='path to directory to save results')
    parser.add_argument('--target-column', help='Name of target column in CSV files', default='steps')
    args = parser.parse_args()
    return args


def init_model(_model, _device, t_dim=500, n_mels=64):
    """
    Args:
        t_dim:
        n_mels: number of mel-bins of spectogram
        _model: Lowercase string, matching the name of the model
        _device: cpu or gpu ?
    Returns: Initialized Model
    """
    m = None

    # -----------------------------------------------------------------------------------------------------------------#
    # DNNs

    if _model == 'dnn':
        m = dnn.DNN(
            input_size=t_dim * n_mels,
            hidden_size=256,
            # n_layers=5,
            # dropout_p=0.2,
            batch_norm=False
        )
    elif _model == 'cdnn':
        m = dnn.CDNN(output_dim=1, sigmoid_output=True)

    # -----------------------------------------------------------------------------------------------------------------#
    # CNNs


    elif _model == 'cnn14':
        m = cnn.Cnn14(output_dim=1, sigmoid_output=True)
    elif _model == 'cnn10':
        m = cnn.Cnn10(output_dim=1, sigmoid_output=True)
    elif _model == 'cnn10m128':
        m = cnn.Cnn10M128(output_dim=1, sigmoid_output=True)
    elif _model == 'cnn6':
        m = cnn.Cnn6(output_dim=1, sigmoid_output=True)
    elif _model == 'resnet38':
        m = cnn.ResNet38(output_dim=1, sigmoid_output=True)
        m.to(_device)
    elif _model == 'resnet54':
        m = cnn.ResNet54(output_dim=1, sigmoid_output=True)
    elif _model == 'resnet22':
        m = cnn.ResNet22(output_dim=1, sigmoid_output=True)
    elif _model == 'psla':
        m = cnn.EffNetAttention(label_dim=1, pretrain=False, b=0, head_num=0, augment=False)


    elif _model == 'specnn':
        m = cnn.CnnSpec(
            tdim=t_dim,
            fdim=n_mels,
            output_dim=1,
            augment=False
        )


    # -----------------------------------------------------------------------------------------------------------------#
    # RNNs

    elif _model == 'gru':
        m = rnn.GRU(
            input_size=n_mels,
            num_layers=5,
            device=_device,
            hidden_size=256,
        )

    elif _model == 'lstm':
        m = rnn.LSTM(
            input_size=n_mels,
            num_layers=5,
            device=_device,
            hidden_size=256,
            batch_norm=False,
            dropout_p=0.2
        )

    # -----------------------------------------------------------------------------------------------------------------#
    # CRNNS

    elif _model == 'cnnlstmv1':
        m = crnn.CNNLSTM1(input_size=128, hidden_size=128, output_size=1, num_layers=4)
    elif _model == 'cnnlstmv2':
        m = crnn.CNNLSTM2(input_size=128, hidden_size=128, output_size=1)
    elif _model == 'cnn10lstm':
        m = crnn.CNN10LSTM(input_size=512, hidden_size=512, output_size=1, num_layers=3)
    elif _model == 'facrnn':
        m = crnn.FACRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2)
    elif _model == 'fdycrnn':
        m = crnn.FDYCRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2)
    elif _model == 'ssedcrnn':
        m = crnn.SSEDCRNN()
    elif _model == 'crnn_dc20':
        m = crnn.CRNN_DC20(n_in_channel=1, n_class=1, n_RNN_cell=128, n_layers_RNN=4, dropout_recurrent=.2)

    # -----------------------------------------------------------------------------------------------------------------#
    # Tformers

    elif _model == 'ast':

        # Experimental Setup from 'AST':
        # optimizer: Adam
        # quote form paper:
        # For balanced set experiments, we use an
        # initial learning rate of 5e-5 and train the model for 25 epochs,
        # the learning rate is cut into half every 5 epoch after the 10th
        # epoch. For full set experiments, we use an initial learning rate
        # of 1e-5 and train the model for 5 epochs, the learning rate is
        # cut into half every epoch after the 2nd epoch.

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 2))

        m = tfr.ASTModel(
            label_dim=1,  # 1 class
            input_fdim=n_mels,
            input_tdim=t_dim,
            imagenet_pretrain = True,
            audioset_pretrain = True,
            #model_size = 'small224',
            verbose=True
        )

    # -----------------------------------------------------------------------------------------------------------------#

    return m


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main():
    args = parse_args()
    device = get_default_device()

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

    # ------------------------------------------------------------------------------------------------------------------
    # DataLoaders :
    # Read data:
    features = pd.read_csv(args.features).set_index(['file', 'start', 'end'])
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), x)
    )
    db_args = {
        'features': features,
        'target_column': args.target_column,
        'transform': audtorch.transforms.RandomCrop(500, axis=-2)
        # https://audtorch.readthedocs.io/en/0.4.1/api-transforms.html
    }
    db_class = CachedDataset

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
    )

    # ------------------------------------------------------------------------------------------------------------------

    # Model :
    model = init_model(args.approach, device, 500, args.nmels)
    if model is None:
        print('Error initializing model! (please check args.approach) Exiting...')
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

    print('=' * 15, 'EVALUATION', '=' * 15)
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


if __name__ == '__main__':
    main()

