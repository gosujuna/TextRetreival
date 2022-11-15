import os
import torch
import numpy as np
import skvideo.io
from tqdm import tqdm

DATASET_ROOT = './data/vatex'

video_dir = os.path.join(DATASET_ROOT, 'vatex_validation')
out_dir = os.path.join(DATASET_ROOT, 'vatex_validation_tensors')
video_names = list(filter(lambda x: '.mp4' in x, os.listdir(video_dir)))


for file in tqdm(video_names):
    video_name = file.split('.')[0]
    video_path = os.path.join(video_dir, file)

    try:
        video_data = skvideo.io.vread(video_path)
    except:
        print(f'Video file failed to be read: {video_name}')
        continue

    try:
        assert video_data.dtype == np.uint8
    except AssertionError:
        print(f'Video not read as byte array: {video_name}')
        video_data = video_data.astype(np.uint8)

    video_data = torch.ByteTensor(video_data)

    success = False
    while not success:
        with open(os.path.join(out_dir, f'{video_name}.pt'), 'wb') as f:
            torch.save(video_data, f)

        with open(os.path.join(out_dir, f'{video_name}.pt'), 'rb') as f:
            loaded_data = torch.load(f)

        if loaded_data.dtype == torch.uint8:
            success = True
        else:
            print(f'Tensor not loaded as byte tensor: {video_name}')
