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

#
from hear21passt.models.preprocess import AugmentMelSTFT
from hear21passt.base import get_basic_model, get_model_passt

from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import matplotlib.pyplot as plt

from tools.utils import (
    evaluate_regression,
    transfer_features,
    model_plot_reg,
    Config,
    MinMaxScaler,
    plot_losses, get_lr, percent_diff, mae_baseline_dynamic, init_logger
)

from tools.datasets import (
    WavDataset, SpecDataset,
)

import models.ffnn as dnn
import models.cnn as cnn
import models.rnn as rnn
import models.crnn as crnn
import models.transformer as tfr

# REQUIRES RUNNING waveforms.py THAT WORKS THE SAME AS melspects.py BUT FOR (WINDOWED) WAVEFORMS

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
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--mixup', default=False, action='store_true')
    parser.add_argument('--loss', default='mse', choices=['mse', 'huber'])
    parser.add_argument('--input-form', default='spec', choices=['spec', 'wav'])
    parser.add_argument('--optimizer', default='Adam',
                        choices=['SGD', 'Adam', 'RMSprop', 'AdamW', 'Adagrad', 'Adadelta'])
    parser.add_argument('--graph', default=False, )
    parser.add_argument('--wdwsize', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nmels', type=int, default=64)
    parser.add_argument('--feat-type', default='wav',
                        choices=['wav', 'melspec', 'chroma_stft', 'chroma_cqt', 'chroma_cens', 'mfcc',
                                 'spectral_contrast', 'poly_features', 'tonnetz'])
    parser.add_argument('--evalmode', default=False, action='store_true')
    parser.add_argument('--intmode', default=False, action='store_true')
    parser.add_argument('--aug', default='None',
                        choices=['sa', 'fa', '0.1', '0.3', '0.7', 'None'])
    args = parser.parse_args()
    return args


def init_model(_model, _device, _input_shape, sample_rate=16000, wdw_size=5, aug='None'):
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

    if aug != 'None':
        logging.info(f'Using augmentation: {aug}')

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
        m = dnn.CDNN(output_dim=1, sigmoid_output=False)
    # -----------------------------------------------------------------------------------------------------------------#
    # CNNs
    elif args.approach == 'wvcnn14':
        m = cnn.WavegramCnn14(
            sample_rate=16000,
            window_size=320,
            hop_size=160,
            mel_bins=64,
            fmin=0,
            fmax=8000,
            classes_num=1,
            sigmoid_output=False,

        )
    elif args.approach == 'wvlmcnn14':
        m = cnn.WavegramLogmelCnn14(
            sample_rate=16000,
            tdim=wdw_size,
            window_size=320,
            mel_bins=64,
            fmin=0,
            fmax=8000,
            classes_num=1,
            sigmoid_output=False,
            augment=aug
        )
    # -----------------------------------------------------------------------------------------------------------------#
    elif _model == 'ast':

        # REQUIRES timm==0.4.5

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
            input_fdim=_input_shape[0],
            input_tdim=_input_shape[1],
            imagenet_pretrain = True,
            audioset_pretrain = False,
            #model_size = 'small224',
            verbose=True
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
            win_length=800,  # 800
            hopsize=320,  # 320
            n_fft=1024,  # 1024
            freqm=48,
            timem=192,  # 192
            htk=False,
            fmin=50,
            fmax=None,  # None
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000  #w 2000
        )
        logging.info('Parametersettings-PaSST: 128, 16000, 800, 320, 1024, 48, 192, 2000')
        # PRETRAINING NET (trained for results)
        net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=1, pretrained=True)
        # pretraining on audioset

        # OUR HPARAMS NET
        #net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=1, pretrained=False, input_fdim=128, input_tdim=500)

        m = tfr.PasstBasicWrapper(mel=mel, net=net, mode="logits")
    elif _model == 'hts':
        # REQUIRES timm==0.4.11 (?)
        import models.config_hts as config  # the input args can be defined here..
        # config args that are being used in the model: just for mel-spec extraction.
        # melspectrogram created in forward function

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
            use_checkpoint=False,
            sigmoid_output=False
        )
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

#
# #################
#


def mixup_collate_fn2(batch):
    x_batch, y_batch = zip(*batch)
    # alpha = float(alpha) # TODO GLOBAL VARIABLE
    alpha = 0.3
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


if __name__ == '__main__':

    # logging.info('Reading Args and Processing Data..')
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

    # Scale steps
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

    logging.info('DONE (t={:0.2f}s).'.format(time.time() - t0))
    # ------------------------------------------------------------------------------------------------------------------
    # DataLoaders :
    feature_type = args.feat_type
    input_shape = None

    logging.info(f'Loading {args.feat_type}-Features.')
    t0 = time.time()

    # using default waveform (to feed into model(s), which can possibly extract further features..)
    # i.e. wvlmcnn14, which uses raw waveform and mel-specs in one forward pass..
    if feature_type == 'wav':
        logging.info('waveform input')
        db_args = {
            'features': features,
            'target_column': args.target_column,
            'transform': audtorch.transforms.RandomCrop(sample_rate * window_size_seconds, axis=-1)
        }
        db_class = WavDataset

        input_shape = sample_rate * window_size_seconds

    # using spectral features to feed into model(s)
    else:
        # stft settings:
        n_fft = 320
        hop_length = 160
        tdim = math.ceil((sample_rate * window_size_seconds - n_fft) / hop_length) + 1

        # frequency dimensions:
        n_mels = 64
        n_mfcc = 7
        n_chroma = 16


        if feature_type == 'chroma_stft':  # todo
            input_shape = (n_chroma,tdim)
        elif feature_type == 'melspec':
            input_shape = (n_mels,tdim)
        elif feature_type == 'chroma_cqt':
            input_shape = (n_chroma,tdim+2)  # ?
        elif feature_type == 'chroma_cens':  # todo
            input_shape = (n_chroma,tdim)
        elif feature_type == 'mfcc':
            input_shape = (n_mfcc, tdim)  # (1, n_mfcc, tdim)
        elif feature_type == 'spectral_contrast':  # todo
            input_shape = (0,tdim)
        elif feature_type == 'poly_features':  # todo
            input_shape = (0,tdim)
        elif feature_type == 'tonnetz':  # todo
            input_shape = (n_chroma,tdim)
        else:
            logging.info(f'Type {feature_type} not defined.')
            exit(-1)
            

        db_args = {
            'features': features,
            'target_column': args.target_column,
            '_type': feature_type,
            'transform': audtorch.transforms.RandomCrop(sample_rate * window_size_seconds, axis=-1),
            'window_size': window_size_seconds,
            'sample_rate': sample_rate,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'n_mfcc': n_mfcc,
            'n_chroma': n_chroma,
        }
        db_class = SpecDataset
    logging.info(f'used shape:{input_shape}')

    train_dataset = db_class(
        df_train.copy(),
        **db_args
    )
    x, y = train_dataset[0]

    logging.info(f'input_shape:{x.shape}')
    # assert input_shape == x.shape, f'Invalid Input shape: {x.shape}'
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

    if args.aug in ['0.1', '0.3', '0.7']:
        alpha = float(args.aug)
        logging.info(f'using mixup @ {alpha}')
        # collatefn = mixup_collate_fn2(alpha=alpha)
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
    )  # sample dimensions: ([1, 1, 80000])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )  # sample dimensions: ([1, 1, 80000])


    logging.info('DONE (t={:0.2f}s).'.format(time.time() - t0))
    # ------------------------------------------------------------------------------------------------------------------

    # Model :
    logging.info('Loading Model..')
    t0 = time.time()

    model = init_model(args.approach, device, input_shape, sample_rate=sample_rate, wdw_size=args.wdwsize, aug=args.aug)
    if model is None:
        logging.info('Error initializing model! (please check args.approach) Exiting...')
        exit(-1)

    criterion = torch.nn.MSELoss()

    if args.state is not None:
        logging.info(f'Using Checkpoint: {args.state}')
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
            patience=5,  # 4
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
                    desc=f'Epoch {epoch + 1}',
                    total=len(train_loader)
            ):
                x = transfer_features(features, device).squeeze(1)
                output = model(x).squeeze(1)
                targets = targets.to(device)

                if args.intmode:
                    output_rounded = output.round()  # round the tensor
                    loss = criterion(output_rounded, targets.float())  # calculate the loss using the rounded output
                elif args.approach == 'ast':
                    loss = criterion(output, targets.half())
                else:
                    loss = criterion(output, targets.float())  # calculate the loss using the original output

                train_loss.append(loss.item())
                if index % 50 == 0:
                    writer.add_scalar('Loss', loss, global_step=epoch * len(train_loader) + index)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # dev set evaluation
            results, targets, outputs, val_loss = evaluate_regression(
                model,
                device,
                dev_loader,
                transfer_features,
                scaler,
                args.approach,
                args.intmode
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
                writer.add_scalar(f'dev/{metric}', results[metric], (epoch + 1) * len(train_loader))
            logging.info(f'Dev results at epoch {epoch + 1}:\n{yaml.dump(results)}\n')

            # update best
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
                    model, device, test_loader, transfer_features, scaler, args.approach, args.intmode
                    # evaluate model with test_loarder
                )
                df_test['predictions'] = outputs
                df_test[args.target_column] = targets
                df_test.reset_index().to_csv(os.path.join(epoch_folder, 'test.csv'), index=False)
                with open(os.path.join(experiment_folder, f'test{epoch+1}.yaml'), 'w') as fp:
                    yaml.dump(test_results, fp)
                logging.info(f'Best test results @ Epoch{epoch + 1} found @ Epoch{best_epoch + 1}\n{yaml.dump(test_results)}')
                elapsed = (time.time() - t0) // 60
                logging.info(f'Elapsed: {elapsed} min')
                logging.info(percent_diff(base=mae_baseline_dynamic(args.data_root), new=test_results['MAE']))

            # LR-Schedule Rule:
            if (epoch <= warmup_threshold) and args.state is None:  # if we use a checkpoint don't warm-up (?)
                # scheduler.step(epoch)
                pass
            else:
                plateau_scheduler.step(results['MAE'])

        logging.info(f'Best dev results found at epoch {best_epoch + 1}:\n{yaml.dump(best_results)}')
        best_results['Epoch'] = best_epoch + 1
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
        writer.close()
    else:
        best_state = torch.load(os.path.join(
            experiment_folder, 'state.pth.tar'))
        logging.info('Training already run')

    torch.save(best_state, os.path.join(experiment_folder, 'state.pth.tar'))

    # ------------------------------------------------------------------------------------------------------------------
    # Test:

    if not os.path.exists(os.path.join(experiment_folder, 'test.yaml')):
        model.load_state_dict(best_state)  # load best checkpoint for evaluation
        scaler = audobject.from_yaml(os.path.join(experiment_folder, 'scaler.yaml'))
        test_results, targets, outputs, _ = evaluate_regression(
            model, device, test_loader, transfer_features, scaler, args.approach,
                args.intmode  # evaluate model with test_loader
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

