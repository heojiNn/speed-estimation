import argparse
import audiofile as af
import numpy as np
import os
import pandas as pd
import torch
import torchaudio
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mel-spectrogram feature extraction')
    parser.add_argument('--src', help='Path to CSV containing timestamps')
    parser.add_argument('--root', help='Path to raw data folder')
    parser.add_argument('--dest', help='Folder to store features in')
    parser.add_argument('--nmels', type=int, default=64)
    parser.add_argument('--sample-rate', default=16000)

    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=320,  # 320
        f_min=50,   # 50 v , 256
        f_max=8000,  # 8000, 2048
        n_mels=args.nmels,  # 64
    )

    filenames = []
    df = pd.read_csv(args.src)

    # This will fail if the following columns are missing
    # Start must be beginning of segment in seconds
    # End must be end of segment in seconds
    df.set_index(['file', 'start', 'end'], inplace=True)
    for counter, (file, start, end) in tqdm.tqdm(
            enumerate(df.index),
            total=len(df),
            desc='Melspects'
    ):
        offset = start
        duration = end - start

        audio, fs = af.read(
            os.path.join(args.root, file),
            always_2d=True,
            offset=offset,
            duration=duration
        )

        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdims=True)
        if fs != args.sample_rate:
            audio = torchaudio.transforms.Resample(fs, args.sample_rate)(torch.from_numpy(audio))
        else:
            audio = torch.from_numpy(audio)

        try:
            logmel = transform(audio)
        except RuntimeError as e:
            print("An error occurred while computing the mel spectrogram:", e)

            # Check for corruptness
            if torch.isnan(audio).any():
                print("The input audio data is corrupted.")

            # Check for reshaping
            elif audio.numel() == 0:
                print("The input audio data has 0 elements.")
            else:
                # Reshape the tensor to have a non-zero number of elements
                audio = audio.reshape(1, -1)
                logmel = transform(audio)


        filename = '{:012}.npy'.format(counter)
        np.save(os.path.join(args.dest, filename), logmel)
        filenames.append(filename)

    features = pd.DataFrame(
        data=filenames,
        index=df.index,
        columns=['features']
    )
    features.reset_index().to_csv(os.path.join(args.dest, 'features.csv'), index=False)
