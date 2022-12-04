import os
from utils.device import DEVICE
from data.vatex import Vatex
from torch.utils.data import DataLoader
from models.video_text_match import VideoTextMatch
from tqdm import tqdm
import torch
from datetime import datetime
import argparse
import yaml

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
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('checkpoint_dir', metavar='C', type=str, help='the path to checkpoint directory containing the config file')
    parser.add_argument('--cfg', type=str, help='the name of the config file containing hyperparameters')
    parser.add_argument('--rsm', type=str, help='the path to the pth file to resume training from', required=False)
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    cfg = args.cfg
    rsm_pth = args.rsm

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
    train_loader = DataLoader(vatex_train, batch_size=train_cfg['batch_size'], collate_fn=vatex_train.collate_fn, shuffle=True)
    eval_loader = DataLoader(vatex_eval, batch_size=eval_cfg['batch_size'], collate_fn=vatex_eval.collate_fn, shuffle=True)

    model = VideoTextMatch(vatex_train.vocab_size(), video_encoder_cfg, text_encoder_cfg, out_dims).to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0, verbose=True)
    epochs = 0
    best_loss = float('inf')
    start_time = datetime.now()
    if rsm_pth:
        print("resuming training from...", rsm_pth)
        checkpoint = torch.load(rsm_pth)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print('previously trained model weights loaded...')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('previously trained optimizer loaded...')
            epochs = checkpoint['epoch']
            best_loss = checkpoint['loss']
            print('best loss was: ', best_loss)
        else:
            model.load_state_dict(checkpoint)
            print('previously trained model weights loaded...')
    else:
        print("starting new training...")
    for i in range(epochs, num_epochs):
        train_loss = train(train_loader, model, optimizer, accumulate_every=accumulate_every)
        print(f'Epoch: {i}\tAvg train loss: {train_loss/len(vatex_train)}')
        eval_loss = eval(eval_loader, model)
        print(f'Epoch: {i}\tAvg eval loss: {eval_loss/len(vatex_eval)}')

        if eval_loss < best_loss:
            best_loss = eval_loss
            with open(os.path.join(checkpoint_dir, f'{int(round(start_time.timestamp()))}_video_text_match_best.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)

        with open(os.path.join(checkpoint_dir, f'{int(round(start_time.timestamp()))}_video_text_match_latest.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open(os.path.join(checkpoint_dir,f'resume_training.pth'),'wb') as f:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                }, f)

        scheduler.step()

    print(f'Best eval loss: {best_loss}')
