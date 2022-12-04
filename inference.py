import argparse
from collections import OrderedDict
import pickle
import nltk
import os
import numpy as np
import torch
from tqdm import tqdm

import yaml
from data.vatex import Vatex

from models.video_text_match import VideoTextMatch
from utils.device import DEVICE

DATASET_ROOT = './data/vatex'

class VideoTextInference:
    def __init__(self, checkpoint_dir, cfg, weights):
        with open(os.path.join(checkpoint_dir, cfg)) as fp:
            cfg = yaml.safe_load(fp)

        out_dims = cfg['model']['out_dims']

        video_encoder_cfg = cfg['model']['video_encoder']
        text_encoder_cfg = cfg['model']['text_encoder']

        train_cfg = cfg['dataset']['train']
        eval_cfg = cfg['dataset']['eval']
        token_count_thresh = cfg['dataset']['token_count_thresh']

        vatex_train = Vatex(is_train=True, num_captions=train_cfg['num_captions'], token_count_thresh=token_count_thresh)
        vatex_eval = Vatex(is_train=False, num_captions=eval_cfg['num_captions'], token_count_thresh=token_count_thresh)

        self.token2index = vatex_eval.token2index

        self.model = VideoTextMatch(vatex_eval.vocab_size(), video_encoder_cfg, text_encoder_cfg, out_dims).to(DEVICE)

        with open(os.path.join(checkpoint_dir, weights), 'rb') as f:
            model.load_state_dict(torch.load(f))
        model = model.eval()

    def get_video_features(self, video_dir):
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

            video_data = torch.FloatTensor(video_data)

            with torch.no_grad():
                video_features = self.model.get_video_features(video_data)

            with open(os.path.join(video_dir, f'{video_name}.pt'), 'wb') as f:
                torch.save(video_features, f)

    def match(self, video_dir, input_text):
        tokenized_text = nltk.tokenize.word_tokenize(input_text.lower())

        text_indices = []
        text_indices.append(self.token2index['</START>'])
        for token in tokenized_text:
            if token in self.token2index:
                text_indices.append(self.token2index[token])
            else:
                text_indices.append(self.token2index["</UNK>"])
        text_indices.append(self.token2index['</END>'])

        tokenized_text_tensor = torch.LongTensor(text_indices)

        with torch.no_grad():
            text_features = self.model.get_text_features(input_text)

            video_features = list(filter(lambda x: '.pt' in x, os.listdir(video_dir)))
            scores = []
            for file in tqdm(video_features):
                video_name = file.split('.')[0]
                video_path = os.path.join(video_dir, file)

                with open(video_path, 'rb') as f:
                    video_features = torch.load(f)

                scores.append(torch.nn.functional.cosine_similarity(video_features, text_features).item())

        best_match = video_features[np.argmax(scores)].split('.')[0]
        return best_match

if __name__ == '__main__':
    inference = VideoTextInference('./checkpoints/video_text_match/c4', 'video_text_match_c4.yml', '1670030214_video_text_match_latest.pth')
    inference.get_video_features(VIDEO_DIRECTORY_HERE)
    best_match = inference.get_text_features(VIDEO_DIRECTORY_HERE, INPUT_TEXT_HERE)
    print(best_match)