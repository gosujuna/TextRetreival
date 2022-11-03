from collections import OrderedDict
import json
import os
import nltk
import torch
import skvideo.io  
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as tv_t

DATASET_ROOT = './data/vatex'

def collate_fn(batch, padding_value=0):
    # batch is a list of tuples of the form (video_tensor, caption_tensor, caption) where
    # video_tensor has shape (L*, C, H, W), caption_tensor has shape (L*,), and caption is just a string
    # where L* is an arbitrary sequence length
    #
    # This function should return a 3-tuple of the form 
    # (batched_video_tensors, batched_caption_tensors, caption_list)
    #
    # batched_video_tensors has shape (batch_size, L_max, C, H, W)
    # caption_tensor has shape (batch_size, L_max)
    # where L_max is the longest sequence in the batch

    collated = []

    for i in range(len(batch[0])):
        c_x = [x[i] for x in batch]
        if i == 2:
            c_x = default_collate(c_x)
        else:
            c_x = pad_sequence(c_x, batch_first=True, padding_value=padding_value)

        collated.append(c_x)
    return tuple(collated)

class Vatex(Dataset):
    def __init__(self):
        # TODO: make these configurable
        height = 240
        width = 240

        # set collate_fn
        self.collate_fn = collate_fn

        # load filepaths and read annotations file
        # TODO: allow for selection between train/validation sets
        self.video_dir = os.path.join(DATASET_ROOT, 'vatex_training')
        self.ann_path = os.path.join(DATASET_ROOT, 'vatex_training_v1.0.json')

        video_names = list(filter(lambda x: '.mp4' in x, os.listdir(self.video_dir)))

        with open(self.ann_path, encoding='utf_8') as f:
            anns = json.load(f)

            # delete chinese captions since we don't need them
            for entry in anns:
                del entry['chCap']

            # filter out entries that don't correspond to a local video file (need to do b/c some of the videos are no longer available)
            videoIds = set(map(lambda x: x.split('.')[0], video_names))
            anns = list(filter(lambda x: x['videoID'] in videoIds, anns))
            self.anns = anns

        # generate vocab from captions
        self.unknown_token = '</UNK>'
        self.start_token = '</START>'
        self.end_token = '</END>'

        self.token_dict = OrderedDict({
            '': -1,
            self.unknown_token: -1,
            self.start_token: -1,
            self.end_token: -1,
        })

        for entry in self.anns:
            for caption in entry['enCap']:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                for token in tokens:
                    if token not in self.token_dict:
                        self.token_dict[token] = 1
                    else:
                        self.token_dict[token] += 1
        # TODO: filter out uncommon tokens from vocab
        # TODO: save/load token counts to file

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
        
        return (videodata, tokenized_caption_tensor, caption)

if __name__ == '__main__':
    vatex = Vatex()
    loader = DataLoader(vatex, batch_size=20, collate_fn=vatex.collate_fn)
    _ = next(iter(loader))