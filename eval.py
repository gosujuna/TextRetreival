import os
from models.video_text_match import VideoTextMatch
from data.vatex import Vatex
from utils.device import DEVICE
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import yaml
import argparse

def topk(video_features, text_features, k=(1, 5, 10)):
    similarity = (video_features @ torch.transpose(text_features, 0, 1)).to('cpu')

    topk, indices = torch.topk(similarity, k=k[-1], dim=1, largest=True, sorted=True)
    idx = torch.unsqueeze(torch.arange(indices.shape[0]), 1)

    accuracies = []
    for k_val in k:
        top_k_val = indices[:, 0:k_val]
        top_k_val = torch.any(top_k_val == idx, dim=1).int()
        k_accuracy = (torch.sum(top_k_val) / len(top_k_val)).item()
        accuracies.append(k_accuracy)

    return accuracies, indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('checkpoint_dir', metavar='N', type=str, help='the path to checkpoint directory containing the config file')
    parser.add_argument('--cfg', type=str, help='the name of the config file containing hyperparameters')
    parser.add_argument('--weights', type=str, help='the name of the file containing the weights to load')

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    cfg = args.cfg
    weights = args.weights

    with open(os.path.join(checkpoint_dir, cfg)) as fp:
        cfg = yaml.safe_load(fp)

    lr = cfg['lr']
    num_epochs = cfg['num_epochs']
    accumulate_every = cfg['accumulate_every']

    out_dims = cfg['model']['out_dims']

    token_count_thresh = cfg['dataset']['token_count_thresh']

    train_cfg = cfg['dataset']['train']
    eval_cfg = cfg['dataset']['eval']

    video_encoder_cfg = cfg['model']['video_encoder']
    text_encoder_cfg = cfg['model']['text_encoder']

    vatex_train = Vatex(is_train=True, num_captions=train_cfg['num_captions'], token_count_thresh=token_count_thresh)
    vatex_eval = Vatex(is_train=False, num_captions=eval_cfg['num_captions'], token_count_thresh=token_count_thresh)
    train_loader = DataLoader(vatex_train, batch_size=train_cfg['batch_size'], collate_fn=vatex_eval.collate_fn, shuffle=True)
    eval_loader = DataLoader(vatex_eval, batch_size=eval_cfg['batch_size'], collate_fn=vatex_eval.collate_fn, shuffle=True)
    model = VideoTextMatch(vatex_train.vocab_size(), video_encoder_cfg, text_encoder_cfg, out_dims).to(DEVICE)

    with open(os.path.join(checkpoint_dir, weights), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model = model.eval()

    video_features = []
    text_features = []
    captions = []
    video_ids = []

    with torch.no_grad():
        for video, text, caption, video_id in tqdm(eval_loader):
            video = video.to(DEVICE)
            text = text.to(DEVICE)

            video_out, text_out = model(video, text)

            video_features.append(video_out)
            text_features.append(text_out)
            captions.extend(caption)
            video_ids.extend(video_id)

    video_features = torch.cat(video_features)
    text_features = torch.cat(text_features)

    k = (int(.01 * len(vatex_eval)), int(.10 * len(vatex_eval)), int(.25 * len(vatex_eval)))
    #k = (1, 10, 50)

    accuracies, matches = topk(video_features, text_features, k)
    print(accuracies)

    for i in range(5):
        print(f'Example {i+1}')
        caption_idx = matches[i, :5]
        print(video_ids[i])
        for j in caption_idx:
            print(captions[j])