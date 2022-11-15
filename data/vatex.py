from collections import OrderedDict
import json
import os
import nltk
import torch
import pickle
import skvideo.io  
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from utils.device import DEVICE

DATASET_ROOT = './data/vatex'

def collate_fn(batch, padding_value=0):
    # batch is a list of tuples of the form (video_tensor, caption_tensor, caption, videoID) where
    # video_tensor has shape (L*, C, H, W), caption_tensor has shape (L*,), and caption is just a string
    # where L* is an arbitrary sequence length
    #
    # This function should return a 4-tuple of the form 
    # (batched_video_tensors, batched_caption_tensors, caption_list, videoID_list)
    #
    # batched_video_tensors has shape (batch_size, L_max, C, H, W)
    # caption_tensor has shape (batch_size, L_max)
    # where L_max is the longest sequence in the batch

    collated = []

    for i in range(len(batch[0])):
        c_x = [x[i] for x in batch]
        if i != 0 and i != 1:
            c_x = default_collate(c_x)
        else:
            c_x = pad_sequence(c_x, batch_first=True, padding_value=padding_value)

        collated.append(c_x)
    return tuple(collated)

class Vatex(Dataset):
    def __init__(self, is_train=True, token_count_thresh=10, num_captions=10):
        # TODO: make these configurable
        height = 224
        width = 224

        self.num_captions = num_captions

        # set collate_fn
        self.collate_fn = collate_fn

        # load filepaths and read annotations file
        self.video_dir, self.tensor_dir, self.anns = read_annotations(is_train=is_train)

        # load vocab file or generate it if it does not exist
        token_dict = generate_vocab(self.num_captions)

        # filter out uncommon tokens from vocab
        unknown_token = '</UNK>'
        start_token = '</START>'
        end_token = '</END>'

        self.token_dict = OrderedDict({
            '': -1,
            unknown_token: -1,
            start_token: -1,
            end_token: -1,
        })

        self.token_dict.update(
            OrderedDict([(k, token_dict[k]) for k in sorted(token_dict.keys())
                         if token_dict[k] > token_count_thresh])
        )

        self.token2index = OrderedDict([(tkn, ii) for (ii, tkn) in enumerate(self.token_dict.keys())])
        self.index2token = OrderedDict([x for x in enumerate(self.token_dict.keys())])

        # TODO: more transforms?
        self.transforms = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.anns) * self.num_captions

    def __getitem__(self, index):
        # for each video, we generate num_captions number of true AND false pairings

        idx = index // self.num_captions

        entry = self.anns[idx]
        videoId = entry['videoID']

        if index % self.num_captions < self.num_captions:
            caption = entry['enCap'][index % self.num_captions]

        tokenized_caption = nltk.tokenize.word_tokenize(caption.lower())

        caption_indices = []
        caption_indices.append(self.token2index['</START>'])
        for token in tokenized_caption:
            if token in self.token2index:
                caption_indices.append(self.token2index[token])
            else:
                caption_indices.append(self.token2index["</UNK>"])
        caption_indices.append(self.token2index['</END>'])

        tokenized_caption_tensor = torch.LongTensor(caption_indices)

        tensor_path = os.path.join(self.tensor_dir, f'{videoId}.pt')
        
        # attempt to load tensor from disk while checking for errors
        success = False
        attempts = 0
        while not success:
            with open(tensor_path, 'rb') as f:
                video_data = torch.load(f).to(DEVICE) 

            if video_data.dtype == torch.uint8:
                video_data = video_data.type(torch.FloatTensor)
                success = True
            else:
                print(videoId)
                video_path = os.path.join(self.video_dir, f'{videoId}.mp4')
                video_data = skvideo.io.vread(video_path)
                if video_data.dtype == np.uint8:
                    success = True
                    video_data = torch.FloatTensor(video_data).to(DEVICE)
                    with open(os.path.join(tensor_path), 'wb') as f:
                        torch.save(video_data, f)
            attempts += 1
            if attempts > 5:
                print(f'Failed to load video with id: {videoId}')
                raise Exception 

        # video_data has shape (N frames, H, W, C)
        video_data = torch.permute(video_data, (0, 3, 1, 2))
        video_data = torch.nan_to_num(self.transforms(video_data), nan=0)
        
        return (video_data, tokenized_caption_tensor, caption, videoId)

    def vocab_size(self):
        return len(self.token_dict)

# vocab is always generated off of the training data; during validation, tokens that do not appear in the training data will get mapped to </UNK>
def generate_vocab(num_captions):
    _, __, anns = read_annotations(is_train=True)

    # generate vocab from captions
    token_dict = OrderedDict()

    for entry in anns:
        for i in range(num_captions):
            caption = entry['enCap'][i]
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            for token in tokens:
                if token not in token_dict:
                    token_dict[token] = 1
                else:
                    token_dict[token] += 1
    
    # save token_counts to file
    with open(os.path.join(DATASET_ROOT, 'token_counts.dict'), 'wb') as f:
        pickle.dump(token_dict, f)

    return token_dict

# Reads the training or validation json data and returns the filtered and deserialized annotations
def read_annotations(is_train=True):
    if is_train:
        video_dir = os.path.join(DATASET_ROOT, 'vatex_training')
        tensor_dir = os.path.join(DATASET_ROOT, 'vatex_training_tensors')
        ann_path = os.path.join(DATASET_ROOT, 'vatex_training_v1.0.json')
    else:
        video_dir = os.path.join(DATASET_ROOT, 'vatex_validation')
        tensor_dir = os.path.join(DATASET_ROOT, 'vatex_validation_tensors')
        ann_path = os.path.join(DATASET_ROOT, 'vatex_validation_v1.0.json')

    subset = 1000 if is_train else 100
    video_names = list(filter(lambda x: '.pt' in x, os.listdir(tensor_dir)))[:]

    with open(ann_path, encoding='utf_8') as f:
        anns = json.load(f)

        # delete chinese captions since we don't need them
        for entry in anns:
            del entry['chCap']

        # filter out entries that don't correspond to a local video file (need to do b/c some of the videos are no longer available)
        videoIds = set(map(lambda x: x.split('.')[0], video_names))
        anns = list(filter(lambda x: x['videoID'] in videoIds, anns))
        return video_dir, tensor_dir, anns

if __name__ == '__main__':
    vatex_train = Vatex(is_train=True, num_captions=5)
    vatex_validation = Vatex(is_train=False, num_captions=1)
    print(len(vatex_train), len(vatex_validation))
    print(len(vatex_train.token_dict))
    train_loader = DataLoader(vatex_train, batch_size=20, collate_fn=vatex_train.collate_fn)
    # validation_loader = DataLoader(vatex_validation, batch_size=20, collate_fn=vatex_validation.collate_fn)
    data = next(iter(train_loader))
    print(data[0].shape)
    print(data[1].shape)
    print(data[2])
    print(data[3])
    print(data[4])

    # data = next(iter(validation_loader))
    # print(data[0].shape)
    # print(data[1].shape)
    # print(data[2][0])
    # print(data[3][0])