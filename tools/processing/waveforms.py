import argparse
import audiofile as af
import numpy as np
import os
import pandas as pd
import torch
import torchaudio
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Waveform windowing')
    parser.add_argument('--src', help='Path to CSV containing timestamps')
    parser.add_argument('--root', help='Path to raw data folder')
    parser.add_argument('--dest', help='Folder to store features in')
    parser.add_argument('--sample-rate', default=16000)

    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    filenames = []
    df = pd.read_csv(args.src)

    # This will fail if the following columns are missing
    # Start must be beginning of segment in seconds
    # End must be end of segment in seconds
    df.set_index(['file', 'start', 'end'], inplace=True)
    for counter, (file, start, end) in tqdm.tqdm(
            enumerate(df.index),
            total=len(df),
            desc='Waveforms'
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

        filename = '{:012}.npy'.format(counter)
        np.save(os.path.join(args.dest, filename), audio)
        filenames.append(filename)

    features = pd.DataFrame(
        data=filenames,
        index=df.index,
        columns=['features']
    )
    features.reset_index().to_csv(os.path.join(args.dest, 'features.csv'), index=False)
