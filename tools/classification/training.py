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

from models import crnn
from tools.utils import (
    evaluate_sed,
    transfer_features,
    model_plot_reg,
    Config,
    MinMaxScaler,
    plot_losses, get_lr, percent_diff, mae_baseline_dynamic, init_logger
)

from tools.datasets import (
    WavDataset, SpecDataset, CachedDataset,
)

import models.ffnn as dnn
import models.cnn as cnn
import models.transformer as tfr


def parse_args():
    """Returns: cl args"""
    parser = argparse.ArgumentParser('KIRun Speed Training')
    parser.add_argument('--data-root', help='Path data has been extracted', required=True)
    parser.add_argument('--results-root', help='Path where results are to be stored', required=True)
    # see preprocess.py for further information...
    parser.add_argument('--target-column', help='Name of target column in CSV files', default='activities',
                        choices=['activities', 'active'])  # 1st: concatenated window, 2nd: default (small) sed window
    parser.add_argument('--features', help='Path to features', required=True)
    parser.add_argument('--device', help='CUDA-enabled device to use for training', default='cuda:0')
    parser.add_argument('--state', help='Optional initial state')
    parser.add_argument('--approach', default='cnn10')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--minlr', help='Minimal Learning Rate for (down-)scheduler', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--loss', default='mse', choices=['mse', 'huber'])
    parser.add_argument('--input-form', default='spec', choices=['spec', 'wav'])
    parser.add_argument('--optimizer', default='Adam',
                        choices=['SGD', 'Adam', 'RMSprop', 'AdamW', 'Adagrad', 'Adadelta'])
    parser.add_argument('--graph', default=False)
    parser.add_argument('--nclasses', help='Outputdimension', type=int, default=25)
    parser.add_argument('--wdwsize', type=float, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nmels', type=int, default=64)
    parser.add_argument('--feat-type', default='melspec',
                        choices=['wav', 'melspec', 'chroma_stft', 'chroma_cqt', 'chroma_cens', 'mfcc',
                                 'spectral_contrast', 'poly_features', 'tonnetz'])
    parser.add_argument('--evalmode', default=False, action='store_true')
    args = parser.parse_args()
    return args


global swin


def init_model(_model, _device, _input_shape, sample_rate=16000, wdw_size=5, n_classes=25):
    m = None
    global swin
    swin = False

    if _input_shape is None:  # e.g. no spectral features -> default shape: (SR, WDW), i.e. (16000, .2)
        _input_shape = sample_rate, wdw_size

    if _model == '' or _model == 'dnn':
        m = dnn.DNN(
            input_size=_input_shape[0] * _input_shape[1],  # only works for melspectrogram
            hidden_size=256,
            batch_norm=False,
            augment=False,
            out_dim=n_classes,
            sigmoid_output=True
        )
    elif _model == 'cnn10':
        m = cnn.Cnn10(output_dim=n_classes, sigmoid_output=True)
    elif args.approach == 'wvlmcnn14':
        m = cnn.WavegramLogmelCnn14(
            sample_rate=16000,
            window_size=320,
            mel_bins=64,
            fmin=0,
            fmax=8000,
            classes_num=n_classes,
            sigmoid_output=True,
        )
    elif _model == 'cnn14max':
        m = cnn.Cnn14_DecisionLevelMax(classes_num=n_classes)
    elif _model == 'psla':
        m = cnn.EffNetAttention(label_dim=n_classes, pretrain=True, b=0, head_num=0, augment=False)
    elif _model == 'facrnn':
        m = crnn.FACRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2,
                        wav=False, concat=False, n_class=n_classes)
    elif _model == 'fdycrnn':
        m = crnn.FDYCRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2, n_class=n_classes)

    elif _model == 'hts':
        # REQUIRES timm==0.4.11 (?)
        import models.config_hts as config  # the input args can be defined here..
        # config args that are being used in the model: just for mel-spec extraction.
        # TODO: either export config to code or export code into one combined config applicable for all models...

        # melspectrogram created in forward function

        m = tfr.HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=n_classes,
            window_size=config.htsat_window_size,
            config=config,
            depths=config.htsat_depth,
            embed_dim=config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head,
            use_checkpoint=False
        )

    elif _model == 'passt':

        # REQUIRES timm==0.4.12
        # Experimental Setup from 'Efficient Training of Audio Transformers with Patchout':
        # quote form paper:
        # We use the AdamW optimizer with weight decay of 10e-4, with a
        # maximum learning rate of 10e-5. We use a linear learning rate
        # decay from epoch 50 to 100, dropping the learning rate to 10e-7
        # and fine-tune the model for a further 20 epochs.

        mel = AugmentMelSTFT(
            n_mels=128,  # 128
            sr=16000,
            win_length=320,  # 800
            hopsize=160,
            n_fft=320,  # 1024
            freqm=48,
            timem=192,  # 192, consider changing (timemasking might [completely] mask steps)
            htk=False,
            fmin=50,
            fmax=None,  # None
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000
        )

        # PRETRAINING NET
        net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=n_classes,
                              pretrained=True)  # pretraining on audioset

        # HPARAMS NET
        # net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=n_classes,   # 25 classes for 5 s inpt, 200 ms actives
        #                        pretrained=False, input_fdim=128, input_tdim=500)

        m = tfr.PasstBasicWrapper(mel=mel, net=net, mode="logits", prespec=True)

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


def warm_up_cosine(epoch, threshold=3):
    """The learning rate is smoothly increased from 0 to its normal value using a cosine function."""
    return 0.5 * (1 + math.cos(math.pi * epoch / 5))


if __name__ == '__main__':

    global swin

    t0 = time.time()
    args = parse_args()

    # Init logging:
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.results_root, f'{timestamp}_logs')
    init_logger(log_dir, filemode='w')
    logging.info(args)

    device = args.device
    epochs = args.epochs
    save_interval = args.save_interval
    window_size_seconds = args.wdwsize  # seconds
    sample_rate = 16000
    tdim = int(window_size_seconds * 100)  # only works on our
    crop = int(sample_rate * args.wdwsize)

    # Configure meta settings:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    experiment_folder = args.results_root
    os.makedirs(experiment_folder, exist_ok=True)

    # Read data:
    df_train = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    df_train.set_index(['file', 'start', 'end'], inplace=True)
    df_dev = pd.read_csv(os.path.join(args.data_root, 'devel.csv'))
    df_dev.set_index(['file', 'start', 'end'], inplace=True)
    df_test = pd.read_csv(os.path.join(args.data_root, 'test.csv'))
    df_test.set_index(['file', 'start', 'end'], inplace=True)

    # Read data:
    features = pd.read_csv(args.features).set_index(['file', 'start', 'end'])
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(args.features), x)
    )

    logging.info('DONE (t={:0.2f}s).'.format(time.time() - t0))
    # ------------------------------------------------------------------------------------------------------------------
    # DataLoaders :
    feature_type = args.feat_type
    input_shape = None

    logging.info(f'Loading {args.feat_type}-Features.')
    t0 = time.time()

    # using default waveform (to feed into model(s), which can possibly extract further features..)
    # --feat-type wav (for example for waveform input)
    # i.e. wvlmcnn14, which uses raw waveform and mel-specs in one forward pass..
    if feature_type == 'melspec':
        db_args = {
            'features': features,
            'target_column': args.target_column,
            'transform': audtorch.transforms.RandomCrop(tdim, axis=-2)
            # https://audtorch.readthedocs.io/en/0.4.1/api-transforms.html
        }
        db_class = CachedDataset
    elif feature_type == 'wav':
        logging.info('waveform input')
        db_args = {
            'features': features,
            'target_column': args.target_column,
            'transform': audtorch.transforms.RandomCrop(crop, axis=-1)
        }
        db_class = WavDataset
    # using spectral features to feed into model(s)
    else:
        n_fft = 320
        hop_length = 160
        n_mfcc = 7
        n_chroma = 12
        signal_length = sample_rate * window_size_seconds
        tdim = math.floor((signal_length - n_fft) / hop_length) + 1

        print(tdim)

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
            logging.info(f'Type {feature_type} not defined.')
            exit(-1)

        db_args = {
            'features': features,
            'target_column': args.target_column,
            '_type': feature_type,
            'transform': audtorch.transforms.RandomCrop(500, axis=-1),
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
    input_shape = x.shape

    logging.info(f'input_shape:{x.shape}')
    logging.info('DONE (t={:0.2f}s).'.format(time.time() - t0))
    logging.info('Loading Dataset and Trainloaders..')
    t0 = time.time()

    dev_dataset = db_class(
        df_dev.copy(),
        **db_args
    )

    test_dataset = db_class(
        df_test.copy(),
        **db_args
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False
    )
    # sample dimensions:
    #   torch.Size([32, 1, 3200])
    # logging.info('train-sample-shape {}'.format(next(iter(train_loader))[0].shape))

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False
    )  # sample dimensions: ([1, 1, 80000])
    # print(len(dev_loader))
    # exit()

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False
    )  # sample dimensions: ([1, 1, 80000])

    logging.info('DONE (t={:0.2f}s).'.format(time.time() - t0))
    # ------------------------------------------------------------------------------------------------------------------

    # Model :
    logging.info('Loading Model..')
    t0 = time.time()
    # inputshape is None for melspec, and set for other...
    model = init_model(args.approach, device, input_shape, sample_rate=sample_rate,
                       wdw_size=args.wdwsize, n_classes=args.nclasses)
    if model is None:
        logging.info('Error initializing model! (please check args.approach) Exiting...')
        exit(-1)

    criterion = torch.nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    if args.state is not None:
        logging.info('Using Checkpoint..')
        initial_state = torch.load(args.state)
        model.load_state_dict(
            initial_state,
            strict=False
        )
    logging.info('DONE (t={:0.2f}s).'.format(time.time() - t0))

    # ------------------------------------------------------------------------------------------------------------------
    # Train/Eval Loop :

    if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):
        with open(os.path.join(experiment_folder, 'hparams.yaml'), 'w') as fp:
            yaml.dump(vars(args), fp)

        writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))
        optimizer = init_optimizer(args.optimizer, model, args.learning_rate)
        warmup_threshold = 3  # epochs
        # WARM-UP
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warm_up_cosine
        )
        # COOL-DOWN
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,  # new_lr = 0.001 * factor
            patience=3,
            min_lr=args.minlr
        )
        max_metric = 1e32
        best_epoch = 0
        best_state = None
        best_results = None
        train_loss = []
        train_losses = []
        val_losses = []

        logging.info('Starting Training..')
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
                    desc=f'Epoch {epoch}',
                    total=len(train_loader)
            ):
                x = transfer_features(features, device)# .squeeze(1)  # torch.Size([25, 3200])
                # if args.approach == 'passt':
                #     x = x.squeeze(1)
                if args.feat_type == 'wav':
                    if x.ndim == 3:
                        x = x.squeeze(1)
                output = model(x).squeeze(1)

                targets = targets.to(device)
                loss = criterion(output, targets.float())
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            results, targets, outputs, val_loss = evaluate_sed(
                model,
                device,
                dev_loader,
                args.approach,
                args.nclasses
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

            results['PRECISION'] = float(results['PRECISION'])
            results['RECALL'] = float(results['RECALL'])
            results['F1'] = float(results['F1'])

            for metric in results.keys():
                writer.add_scalar(f'dev/{metric}', results[metric], (epoch + 1) * len(train_loader))

            logging.info(f'Dev results at epoch {epoch + 1}:\n{yaml.dump(results)}\n')

            # store one initially
            if epoch == 0:
                max_metric = results['ACC']
                best_epoch = epoch
                best_state = model.cpu().state_dict()
                best_results = results.copy()

            # update best
            if results['ACC'] > max_metric:
                max_metric = results['ACC']
                best_epoch = epoch
                best_state = model.cpu().state_dict()
                best_results = results.copy()

            # intermediate saving and evaluation (for longer training):
            if save_interval != 0 and (epoch + 1) % save_interval == 0 and save_interval != epochs:
                # logg save
                sv_path = os.path.join(experiment_folder, f'state_ep{epoch + 1}.pth.tar')
                if best_state is not None:
                    torch.save(best_state, sv_path)
                logging.info('Model saved to {}'.format(sv_path))

                # eval without saving results
                model.load_state_dict(best_state)  # load best checkpoint for evaluatio

                test_results, targets, outputs, val_loss = evaluate_sed(
                    model, device, test_loader, args.approach, args.nclasses, eval=True
                )
                # evaluate model with test_loarder
                test_results['PRECISION'] = float(results['PRECISION'])
                test_results['F1'] = float(results['F1'])
                test_results['RECALL'] = float(results['RECALL'])
                logging.info(
                    f'Best test results @ Epoch{epoch + 1} found @ Epoch{best_epoch + 1}\n{yaml.dump(test_results)}')

            # LR-Schedule Rule:
            if (epoch <= warmup_threshold) and args.state is None:  # if we use a checkpoint don't warm-up (?)
                # scheduler.step(epoch)
                pass
            else:
                plateau_scheduler.step(results['ACC'])

        logging.info(f'Best dev results found at epoch {best_epoch + 1}:\n{yaml.dump(best_results)}')
        best_results['Epoch'] = best_epoch + 1
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
        writer.close()
    else:
        best_state = torch.load(os.path.join(
            experiment_folder, 'state.pth.tar'))
        logging.info('Training already run')

    if best_state is not None:
        torch.save(best_state, os.path.join(experiment_folder, 'state.pth.tar'))

    # ------------------------------------------------------------------------------------------------------------------
    # Test:

    if not os.path.exists(os.path.join(experiment_folder, 'test.yaml')):
        model.load_state_dict(best_state)  # load best checkpoint for evaluation

        test_results, targets, outputs, val_loss = evaluate_sed(
            model, device, test_loader, args.approach, args.nclasses  # evaluate model with test_loader
        )
        test_results['PRECISION'] = float(results['PRECISION'])
        test_results['F1'] = float(results['F1'])
        test_results['RECALL'] = float(results['RECALL'])
        logging.info(f'Best test results:\n{yaml.dump(test_results)}')

        np.save(os.path.join(experiment_folder, 'targets.npy'), targets)
        np.save(os.path.join(experiment_folder, 'outputs.npy'), outputs)
        df_test['predictions'] = outputs
        df_test[args.target_column] = targets
        df_test.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
        with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
            yaml.dump(test_results, fp)
        elapsed = int((time.time() - t0) / 60)
        logging.info(f'elapsed time: {elapsed} minutes')
        savepath = os.path.join(experiment_folder, 'losses.png')
        plot_losses(train_losses, val_losses, epochs=args.epochs, savepath=savepath)
    else:
        logging.info('Evaluation already run')
