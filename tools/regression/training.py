import argparse
import math

import audobject
import audtorch
import numpy as np
import os
import pandas as pd
import random

import scipy.special as sc
import torch
import tqdm
import yaml
import time
import logging

from hear21passt.models.preprocess import AugmentMelSTFT
from hear21passt.base import get_basic_model, get_model_passt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import matplotlib.pyplot as plt

import models.EfficientAT
from tools import augmentation
from tools.utils import (
    evaluate_regression,
    transfer_features,
    model_plot_reg,
    Config,
    MinMaxScaler,
    plot_losses, get_lr, percent_diff, mae_baseline_dynamic, init_logger
)

from tools.datasets import (
    CachedDataset
)

import models.ffnn as dnn
import models.cnn as cnn
import models.rnn as rnn
import models.crnn as crnn
import models.transformer as tfr


# #################
# INIT


def parse_args():
    """Returns: cl args"""
    parser = argparse.ArgumentParser('KIRun Speed Training')
    parser.add_argument('--data-root', help='Path data has been extracted', required=True)
    parser.add_argument('--results-root', help='Path where results are to be stored', required=True)
    parser.add_argument('--target-column', help='Name of target column in CSV files', default='steps')
    parser.add_argument('--features', help='Path to features', required=True)
    parser.add_argument('--device', help='CUDA-enabled device to use for training', default='cuda:0')
    parser.add_argument('--state', help='Optional initial state')
    parser.add_argument('--approach', default='cnn10')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--minlr', help='Minimal Learning Rate for (down-)scheduler', type=float, default=1e-5)
    parser.add_argument('--wdwsize', help='Necessary to select correct time dimension', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--loss', default='mse', choices=['mse', 'huber'])
    parser.add_argument('--input-form', default='spec', choices=['spec', 'wav'])
    parser.add_argument('--optimizer', default='Adam',
                        choices=['SGD', 'Adam', 'RMSprop', 'AdamW', 'Adagrad', 'Adadelta'])
    parser.add_argument('--graph', default=False, )
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nmels', type=int, default=64)
    parser.add_argument('--aug', default='None',
                        choices=['sa', 'fa', '0.1', '0.3', '0.7', 'None'])
    args = parser.parse_args()
    return args


def init_model(_model, _device, t_dim=500, n_mels=64, aug='None'):
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
        m = cnn.Cnn14(output_dim=1, sigmoid_output=False)
    elif _model == 'cnn10':
        m = cnn.Cnn10(output_dim=1, sigmoid_output=False, augment=True)
    elif _model == 'cnn10m128':
        m = cnn.Cnn10M128(output_dim=1, sigmoid_output=False)
    elif _model == 'cnn6':
        m = cnn.Cnn6(output_dim=1, sigmoid_output=False)
    elif _model == 'resnet38':
        m = cnn.ResNet38(output_dim=1, sigmoid_output=False)
        m.to(_device)
    elif _model == 'resnet54':
        m = cnn.ResNet54(output_dim=1, sigmoid_output=False)
    elif _model == 'resnet22':
        m = cnn.ResNet22(output_dim=1, sigmoid_output=False)
    elif _model == 'mobilenetv1':
        m = cnn.MobileNetV1()
    elif _model == 'mobilenetv2':
        m = cnn.MobileNetV2()
    elif _model == 'effmobilenet':
        m = models.EfficientAT.get_model(width_mult=1.0, pretrained_name="mn10_as", num_classes=1,
                                         input_dim_t=t_dim, input_dim_f=n_mels)  # mn40_as
    elif _model == 'psla':
        m = cnn.EffNetAttention(label_dim=1, pretrain=True, b=0, head_num=0, augment=aug)
    elif _model == 'specnn':
        m = cnn.CnnSpec(tdim=t_dim, fdim=n_mels, output_dim=1, augment=False)

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
        m = crnn.FACRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2, n_class=25)
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
            imagenet_pretrain=True,
            audioset_pretrain=False,  # True better, used False for feature comparison ...
            # model_size = 'small224',
            verbose=True
        )

    # -----------------------------------------------------------------------------------------------------------------#

    return m


def init_loss(_loss):
    pass


def init_optimizer(_opt, _model, _lr):
    opt = None
    if _opt == 'SGD':
        opt = torch.optim.SGD(
            _model.parameters(),
            momentum=0.9,
            lr=_lr
        )
    elif _opt == 'Adam':
        opt = torch.optim.Adam(
            _model.parameters(),
            lr=_lr
        )
    elif _opt == 'AdamW':
        opt = torch.optim.AdamW(
            _model.parameters(),
            lr=_lr,
            # weight_decay=10e-4,  # only for passt
            eps=1e-7
        )
    elif _opt == 'RMSprop':
        opt = torch.optim.RMSprop(
            _model.parameters(),
            lr=_lr,
            alpha=.95,
            eps=1e-7
        )
    elif _opt == 'Adadelta':
        opt = torch.optim.Adadelta(
            _model.parameters(),
            lr=_lr,
        )
    elif _opt == 'Adagrad':
        opt = torch.optim.Adagrad(
            _model.parameters(),
            lr=_lr,
        )

    return opt

# #################
#


def mixup_collate_fn2(batch):
    x_batch, y_batch = zip(*batch)
    alpha = 0.3  # test different values # TODO GLOBAL VARIABLE
    batch_size = len(x_batch)
    # shuffle indices to mix random samples:
    indices = torch.randperm(batch_size)
    x_shuffle = [x_batch[i] for i in indices]
    y_shuffle = [y_batch[i] for i in indices]
    # lambda
    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float().to('cpu:0')
    # mixup:
    x_mix = [lam[i] * x_batch[i] + (1 - lam[i]) * x_shuffle[i] for i in range(batch_size)]
    y_mix = [lam[i] * y_batch[i] + (1 - lam[i]) * y_shuffle[i] for i in range(batch_size)]
    return torch.stack(x_mix), torch.stack(y_mix)


#
# #################
#


if __name__ == '__main__':
    args = parse_args()

    # Init logging:
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.results_root, f'{timestamp}_logs')
    init_logger(log_dir, filemode='w')
    logging.info(args)

    device = args.device
    epochs = args.epochs
    save_interval = args.save_interval
    # The length of the audio file in samples = sample_rate x audio duration in seconds = 16000 x 5 = 80000
    # The number of frames in the spectrogram = (audio_length - n_fft) / hop_length + 1 = (80000 - 320) / 160 + 1 = 495
    # ~= 500, using the same settings for stft, we can use [s] * 100 for t_dim in general
    window_size = args.wdwsize * 100  # t_dim of melspect

    # Clear memory cache
    torch.cuda.empty_cache()

    # Configure meta settings:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    experiment_folder = args.results_root
    os.makedirs(experiment_folder, exist_ok=True)

    # Read annotations:
    df_train = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    df_train.set_index(['file', 'start', 'end'], inplace=True)
    df_dev = pd.read_csv(os.path.join(args.data_root, 'devel.csv'))
    df_dev.set_index(['file', 'start', 'end'], inplace=True)
    df_test = pd.read_csv(os.path.join(args.data_root, 'test.csv'))
    df_test.set_index(['file', 'start', 'end'], inplace=True)

    # Scale steps:
    scaler = MinMaxScaler(
        minimum=df_train[args.target_column].min(),
        maximum=df_train[args.target_column].max()
    )
    df_train[args.target_column] = df_train[args.target_column].apply(scaler.encode)
    df_dev[args.target_column] = df_dev[args.target_column].apply(scaler.encode)
    df_test[args.target_column] = df_test[args.target_column].apply(scaler.encode)
    scaler.to_yaml(os.path.join(experiment_folder, 'scaler.yaml'))

    # Read data:
    features = pd.read_csv(args.features).set_index(['file', 'start', 'end'])
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), x)
    )
    db_args = {
        'features': features,
        'target_column': args.target_column,
        'transform': audtorch.transforms.RandomCrop(window_size, axis=-2)
        # https://audtorch.readthedocs.io/en/0.4.1/api-transforms.html
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Model :

    model = init_model(args.approach, device, window_size, args.nmels)
    if model is None:
        print('Error initializing model! (please check args.approach) Exiting...')
        exit(-1)

    model.to_yaml(os.path.join(experiment_folder, 'model.yaml'))

    criterion = torch.nn.MSELoss()
    if args.state is not None:
        logging.info('Loading checkpoint {}'.format(args.state))
        initial_state = torch.load(args.state)
        model.load_state_dict(
            initial_state,
            strict=False
        )

    # ------------------------------------------------------------------------------------------------------------------
    # DataLoaders :

    db_class = CachedDataset
    train_dataset = db_class(
        df_train.copy(),
        **db_args
    )
    x, y = train_dataset[0]

    # do not crop for dev/test
    # not cropping implies different shapes for dev/test loaders -> invalid input sizes for some models
    # _ = db_args.pop('transform')

    dev_dataset = db_class(
        df_dev.copy(),
        **db_args
    )

    test_dataset = db_class(
        df_test.copy(),
        **db_args
    )

    if args.aug in ['0.1', '0.3', '0.7']:
        alpha = float(args.aug)
        logging.info(f'using mixup @ {alpha}')
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                                   num_workers=4, drop_last=False, collate_fn=mixup_collate_fn2)
    else:
        logging.info('not using mixup')
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                                   num_workers=4, drop_last=False)
    # logging.info('train-sample-shape {}'.format(next(iter(train_loader))[0].shape))

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
        drop_last=False
    )  # sample dimensions: torch.Size([1, 1, 2090, 64]) without RandomCrop, else torch.Size([1, 1, 500, 64])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )  # sample dimensions: torch.Size([1, 1, 595, 64]) without RandomCrop, else torch.Size([1, 1, 500, 64])

    # ------------------------------------------------------------------------------------------------------------------

    # Train/Eval Loop :
    if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):
        with open(os.path.join(experiment_folder, 'hparams.yaml'), 'w') as fp:
            yaml.dump(vars(args), fp)

        writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))
        optimizer = init_optimizer(args.optimizer, model, args.learning_rate)

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,  # new_lr = 0.001 * factor
            patience=5,
            min_lr=args.minlr
        )

        max_metric = 1e32
        best_epoch = 0
        best_state = None
        best_results = None
        train_loss = []
        train_losses = []
        val_losses = []
        t0 = time.time()

        for epoch in range(epochs):
            model.to(device)
            model.train()
            epoch_folder = os.path.join(
                experiment_folder,
                f'Epoch_{epoch + 1}'
            )
            os.makedirs(epoch_folder, exist_ok=True)

            for index, (features, targets) in tqdm.tqdm(
                    enumerate(train_loader),
                    desc=f'Epoch {epoch + 1}',
                    total=len(train_loader)
            ):
                output = model(transfer_features(features, device)).squeeze(1)
                if args.approach in ['ast']:
                    x = transfer_features(features, device).squeeze(1)
                    output = model(x).squeeze(1)
                if args.graph:  # display a graph of a model
                    make_dot(output, params=dict(list(model.named_parameters()))). \
                        render(f"{args.approach}_graph",
                               directory=os.path.join(experiment_folder),
                               format="png")
                targets = targets.to(device)  # cuda
                loss = criterion(output, targets.float())  # L(p, y)
                if args.approach == 'ast':
                    loss = criterion(output, targets.half())
                train_loss.append(loss.item())
                if index % 50 == 0:  # tensorboard-logging
                    writer.add_scalar(
                        'Loss',
                        loss,
                        global_step=epoch * len(train_loader) + index
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #
                # END OF BATCH LOOP _|
                #

            # dev set evaluation
            results, targets, outputs, val_loss = evaluate_regression(
                model,
                device,
                dev_loader,
                transfer_features,
                scaler,
                args.approach
            )
            train_losses.append(np.mean(train_loss))
            val_losses.append(np.mean(val_loss))
            df_dev['predictions'] = outputs
            df_dev[args.target_column] = targets
            df_dev.reset_index().to_csv(os.path.join(epoch_folder, 'dev.csv'), index=False)
            # add lr to metric:
            results['lr'] = get_lr(optimizer)
            # add loss to metric
            results['train_loss'] = float(train_losses[epoch])
            results['val_loss'] = float(val_losses[epoch])
            for metric in results.keys():
                # if results[metric] == math.isnan(x):
                #     logging.info('Training Diverged:{}==.nan'.format(results[metric]))
                #     exit(-1)
                writer.add_scalar(f'dev/{metric}', results[metric], (epoch + 1) * len(train_loader))
            logging.info(f'Dev results at epoch {epoch + 1}:\n{yaml.dump(results)}\n')

            # 'best'-policy:
            if results['MAE'] < max_metric:
                max_metric = results['MAE']
                best_epoch = epoch
                best_state = model.cpu().state_dict()
                best_results = results.copy()

            # intermediate saving and evaluation (for longer training):
            if save_interval != 0 and (epoch + 1) % save_interval == 0 and save_interval != epochs:
                # logg save
                sv_path = os.path.join(experiment_folder, f'state_ep{epoch + 1}.pth.tar')
                torch.save(best_state, sv_path)
                logging.info('Model saved to {}'.format(sv_path))
                # eval without saving results
                model.load_state_dict(best_state)  # load best checkpoint for evaluation
                scaler = audobject.from_yaml(os.path.join(experiment_folder, 'scaler.yaml'))
                test_results, targets, outputs, _ = evaluate_regression(
                    model, device, test_loader, transfer_features, scaler, args.approach
                    # evaluate model with test_loarder
                )
                df_test['predictions'] = outputs
                df_test[args.target_column] = targets
                df_test.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
                with open(os.path.join(experiment_folder, f'test{epoch + 1}.yaml'), 'w') as fp:
                    yaml.dump(test_results, fp)
                logging.info(
                    f'Best test results @ Epoch{epoch + 1} found @ Epoch{best_epoch + 1}\n{yaml.dump(test_results)}')
                elapsed = (time.time() - t0) // 60
                logging.info(f'Elapsed: {elapsed} min')
                logging.info(percent_diff(base=mae_baseline_dynamic(args.data_root), new=test_results['MAE']))

            # scheduler downscales lr depending on best MAE results (CC or CCC also candidate)
            plateau_scheduler.step(results['MAE'])

            #
            # END OF EPOCH LOOP _|
            #

        logging.info(f'Best dev results found at epoch {best_epoch + 1}:\n{yaml.dump(best_results)}')
        best_results['Epoch'] = best_epoch + 1
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
        writer.close()
    else:  # name exists
        best_state = torch.load(os.path.join(
            experiment_folder, 'state.pth.tar'))
        logging.info('Training already run')
    # save best checkpoint in the end:
    torch.save(best_state, os.path.join(experiment_folder, 'state.pth.tar'))

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation on Test-Set:

    if not os.path.exists(os.path.join(experiment_folder, 'test.yaml')):
        model.load_state_dict(best_state)  # load best checkpoint for evaluation
        scaler = audobject.from_yaml(os.path.join(experiment_folder, 'scaler.yaml'))
        test_results, targets, outputs, _ = evaluate_regression(
            model, device, test_loader, transfer_features, scaler, args.approach  # evaluate model with test_loarder
        )
        logging.info(f'Best test results:\n{yaml.dump(test_results)}')
        logging.info(percent_diff(base=mae_baseline_dynamic(args.data_root), new=test_results['MAE']))
        np.save(os.path.join(experiment_folder, 'targets.npy'), targets)
        np.save(os.path.join(experiment_folder, 'outputs.npy'), outputs)
        df_test['predictions'] = outputs
        df_test[args.target_column] = targets
        df_test.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
        with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
            yaml.dump(test_results, fp)
    else:
        logging.info('Evaluation already run')

    # ------------------------------------------------------------------------------------------------------------------
    # Final printouts, like time, prediction vs gt and train vs val-loss

    elapsed = int((time.time() - t0) / 60)
    logging.info(f'elapsed time: {elapsed} minutes')

    cfg = Config(
        args.wdwsize,
        args.fold,
        args.epochs,
        args.optimizer,
        args.batch_size,
        args.learning_rate,
        args.approach
    )

    model_plot_reg(experiment_folder + f'/Epoch_{args.epochs}/test.csv', cfg, save=True)
    savepath = os.path.join(experiment_folder, 'losses.png')
    plot_losses(train_losses, val_losses, epochs=args.epochs, savepath=savepath)
