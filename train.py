import os
from utils.device import DEVICE
from data.vatex import Vatex
from torch.utils.data import DataLoader
from models.video_text_match import VideoTextMatch
from tqdm import tqdm
import torch
from datetime import datetime

CHECKPOINT_DIR = './checkpoints/video_text_match'

def compute_loss(video_features, text_features, temperature):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    video_text_similarity = video_features @ torch.transpose(text_features, 0, 1)
    video_video_similarity = video_features @ torch.transpose(video_features, 0, 1)
    text_text_similarity = text_features @ torch.transpose(text_features, 0, 1)
    target = torch.softmax((video_video_similarity + text_text_similarity) / (2 * temperature), dim=1)

    video_loss = criterion(video_text_similarity, target)
    text_loss = criterion(torch.transpose(video_text_similarity, 0, 1), torch.transpose(target, 0, 1))
    return (video_loss + text_loss) / 2


def train(loader, model, optimizer, accumulate_every=64):
    progress_bar = tqdm(loader)
    model = model.train()

    total_loss = 0
    running_loss = 0
    total_examples = 0
    for i, (video, text, caption, videoId) in enumerate(progress_bar):
        video = video.to(DEVICE)
        text = text.to(DEVICE)

        video_out, text_out = model(video, text)

        loss = compute_loss(video_out, text_out, temperature=1)
        loss = loss / accumulate_every
        loss.backward()

        if (i + 1) % accumulate_every == 0 or i == len(progress_bar) - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(loss.item() * accumulate_every)
        running_loss += float(loss.item() * accumulate_every)
        total_examples += video.shape[0]

        progress_bar.set_description_str("Batch: %d, Batch Loss: %.6f, Avg Loss: %.6f" % ((i + 1), float(loss.item() * accumulate_every), (running_loss/total_examples)))

        if i == 0:
            print(video_out, text_out)
    return total_loss

def eval(loader, model):
    progress_bar = tqdm(loader)
    model = model.eval()

    total_loss = 0
    running_loss = 0
    total_examples = 0

    with torch.no_grad():
        for i, (video, text, caption, videoId) in enumerate(progress_bar):
            video = video.to(DEVICE)
            text = text.to(DEVICE)

            video_out, text_out = model(video, text)
            loss = compute_loss(video_out, text_out, temperature=1)

            total_loss += float(loss.item())
            running_loss += float(loss.item())
            total_examples += video.shape[0]
        progress_bar.set_description_str("Batch: %d, Batch Loss: %.6f, Avg Loss: %.6f" % ((i + 1), float(loss.item()), (running_loss/total_examples)))
    return total_loss

if __name__ == '__main__':
    num_epochs = 100
    num_captions = 1
    batch_size = 4
    token_count_thresh = 4

    video_lstm_input_dims = 300
    text_lstm_input_dims = 300
    hidden_dims = 512
    text_lstm_layers = 2
    video_lstm_layers = 1
    fc_dims = 1024
    out_dims = 1024

    lr = .0001

    vatex_train = Vatex(is_train=True, num_captions=num_captions, token_count_thresh=token_count_thresh)
    vatex_eval = Vatex(is_train=False, num_captions=num_captions, token_count_thresh=token_count_thresh)
    train_loader = DataLoader(vatex_train, batch_size=batch_size, collate_fn=vatex_train.collate_fn, shuffle=True)
    eval_loader = DataLoader(vatex_eval, batch_size=batch_size, collate_fn=vatex_eval.collate_fn, shuffle=True)

    model = VideoTextMatch(vatex_train.vocab_size(), text_lstm_input_dims, video_lstm_input_dims, hidden_dims, text_lstm_layers, video_lstm_layers, fc_dims, out_dims).to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95, verbose=True)

    best_loss = float('inf')
    start_time = datetime.now()
    
    for i in range(num_epochs):
        train_loss = train(train_loader, model, optimizer, accumulate_every=16)
        print(f'Epoch: {i}\tAvg train loss: {train_loss/len(vatex_train)}')
        eval_loss = eval(eval_loader, model)
        print(f'Epoch: {i}\tAvg eval loss: {eval_loss/len(vatex_eval)}')

        if eval_loss < best_loss:
            best_loss = eval_loss
            with open(os.path.join(CHECKPOINT_DIR, f'{int(round(start_time.timestamp()))}_video_text_match_best.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)

        with open(os.path.join(CHECKPOINT_DIR, f'{int(round(start_time.timestamp()))}_video_text_match_latest.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)

        scheduler.step()

    print(f'Best eval loss: {best_loss}')
