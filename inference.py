import argparse
from collections import OrderedDict
import pickle
import nltk
import os
import numpy as np
import torch
from tqdm import tqdm
import skvideo.io
import torchvision.transforms as transforms

import yaml
from data.vatex import Vatex

from models.video_text_match import VideoTextMatch
#from utils.device import DEVICE


DEVICE = torch.device('cpu')
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

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        with open(os.path.join(checkpoint_dir, weights), 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        self.model = self.model.eval()

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
            
            video_data = torch.permute(torch.FloatTensor(video_data), (0, 3, 1, 2))
            video_data = torch.unsqueeze(self.transforms(video_data), 0).to(DEVICE)

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

        tokenized_text_tensor = torch.unsqueeze(torch.LongTensor(text_indices), 0).to(DEVICE)
        print(tokenized_text_tensor.shape)

        with torch.no_grad():
            text_features = self.model.get_text_features(tokenized_text_tensor)

            video_list = list(filter(lambda x: '.pt' in x, os.listdir(video_dir)))
            scores = []
            for file in tqdm(video_list):
                video_name = file.split('.')[0]
                video_path = os.path.join(video_dir, file)

                with open(video_path, 'rb') as f:
                    video_features = torch.load(f)

                scores.append(torch.sum(video_features * text_features).item())

        #print(np.array(video_list))
        best_match = np.array(video_list)[np.argsort(scores)]
        return best_match

if __name__ == '__main__':
    video_dir = './TransNetV2/training/BBC_dataset/bbc_01.mp4_segmented_clips'
    inference = VideoTextInference('./checkpoints/video_text_match/c4', 'video_text_match_c4.yml', '1670030214_video_text_match_latest.pth')
    inference.get_video_features(video_dir)
    best_match = inference.match(video_dir, 'Flowers opening up in the sun')
    print(best_match)