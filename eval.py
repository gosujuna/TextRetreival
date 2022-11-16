import os
from models.video_text_match import VideoTextMatch
from data.vatex import Vatex
from utils.device import DEVICE
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

CHECKPOINT_DIR = './checkpoints/video_text_match'

def topk(video_features, text_features, k=(1, 5, 10)):
    similarity = video_features @ torch.transpose(text_features, 0, 1)

    topk, indices = torch.topk(similarity, k=k[-1], dim=1, largest=True, sorted=True)
    idx = torch.unsqueeze(torch.arange(indices.shape[0]), 1).to(DEVICE)

    accuracies = []
    for k_val in k:
        top_k_val = indices[:, 0:k_val]
        top_k_val = torch.any(top_k_val == idx, dim=1).int()
        k_accuracy = (torch.sum(top_k_val) / len(top_k_val)).item()
        accuracies.append(k_accuracy)

    return accuracies, indices


if __name__ == '__main__':
    num_captions = 1
    batch_size = 1
    token_count_thresh = 4

    video_lstm_input_dims = 300
    text_lstm_input_dims = 300
    hidden_dims = 512
    text_lstm_layers = 2
    video_lstm_layers = 1
    fc_dims = 1024
    out_dims = 1024

    vatex_train = Vatex(is_train=True, num_captions=num_captions, token_count_thresh=token_count_thresh)
    vatex_eval = Vatex(is_train=False, num_captions=num_captions, token_count_thresh=token_count_thresh)
    train_loader = DataLoader(vatex_train, batch_size=batch_size, collate_fn=vatex_eval.collate_fn, shuffle=True)
    eval_loader = DataLoader(vatex_eval, batch_size=batch_size, collate_fn=vatex_eval.collate_fn, shuffle=True)
    model = VideoTextMatch(vatex_train.vocab_size(), text_lstm_input_dims, video_lstm_input_dims, hidden_dims, text_lstm_layers, video_lstm_layers, fc_dims, out_dims).to(DEVICE)

    checkpoint = '1668537176_video_text_match_latest.pth'
    with open(os.path.join(CHECKPOINT_DIR, checkpoint), 'rb') as f:
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
            captions.append(caption)
            video_ids.append(video_id)

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