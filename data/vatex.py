from collections import OrderedDict
import json
import os
import nltk
import torch
import pickle
import skvideo.io  
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as tv_t

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
    def __init__(self, is_train=True):
        # TODO: make these configurable
        height = 240
        width = 240

        # set collate_fn
        self.collate_fn = collate_fn

        # load filepaths and read annotations file
        self.video_dir, self.anns = read_annotations(is_train=is_train)

        # load vocab file or generate it if it does not exist
        token_dict_path = os.path.join(DATASET_ROOT, 'token_counts.dict')
        if os.path.isfile(token_dict_path):
            with open(token_dict_path, 'rb') as f:
                self.token_dict = OrderedDict(pickle.load(f))
        else:
            self.token_dict = generate_vocab()
        # TODO: filter out uncommon tokens from vocab

        self.token2index = OrderedDict([(tkn, ii) for (ii, tkn) in enumerate(self.token_dict.keys())])
        self.index2token = OrderedDict([x for x in enumerate(self.token_dict.keys())])

        # TODO: more transforms
        self.transforms = tv_t.Compose([tv_t.Resize((height, width))])
    
    def __len__(self):
        return len(self.anns) * 10

    def __getitem__(self, index):
        # every video has 10 captions, so __getitem__(5) should get the caption at index 5 of the 1st video, __getitem__(15) should get the caption at index 5 of the 2nd video, ...
        idx = index // 10

        entry = self.anns[idx]
        videoId = entry['videoID']
        caption = entry['enCap'][index % 10]

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

        videoPath = os.path.join(self.video_dir, f'{videoId}.mp4')
        videodata = torch.Tensor(skvideo.io.vread(videoPath)) # has shape (N frames, H, W, C)
        videodata = torch.permute(videodata, (0, 3, 1, 2))
        videodata = self.transforms(videodata)
        
        return (videodata, tokenized_caption_tensor, caption, videoId)

# vocab is always generated off of the training data; during validation, tokens that do not appear in the training data will get mapped to </UNK>
def generate_vocab():
    _, anns = read_annotations(is_train=True)

    # generate vocab from captions
    unknown_token = '</UNK>'
    start_token = '</START>'
    end_token = '</END>'

    token_dict = OrderedDict({
        '': -1,
        unknown_token: -1,
        start_token: -1,
        end_token: -1,
    })

    for entry in anns:
        for caption in entry['enCap']:
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
        ann_path = os.path.join(DATASET_ROOT, 'vatex_training_v1.0.json')
    else:
        pass # TODO: validation data

    video_names = list(filter(lambda x: '.mp4' in x, os.listdir(video_dir)))

    with open(ann_path, encoding='utf_8') as f:
        anns = json.load(f)

        # delete chinese captions since we don't need them
        for entry in anns:
            del entry['chCap']

        # filter out entries that don't correspond to a local video file (need to do b/c some of the videos are no longer available)
        videoIds = set(map(lambda x: x.split('.')[0], video_names))
        anns = list(filter(lambda x: x['videoID'] in videoIds, anns))
        return video_dir, anns

if __name__ == '__main__':
    vatex = Vatex()
    print(len(vatex))
    print(len(vatex.token_dict))
    loader = DataLoader(vatex, batch_size=20, collate_fn=vatex.collate_fn)
    data = next(iter(loader))
    print(data[0].shape)
    print(data[1].shape)
    print(data[2][0])
    print(data[3][0])