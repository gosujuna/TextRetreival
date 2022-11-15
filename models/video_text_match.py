import torch.nn as nn
import torchvision
import torch
from torch.utils.data import DataLoader
from data.vatex import Vatex
from utils.device import DEVICE

class VideoEncoder(nn.Module):
    def __init__(self, lstm_input_dims, hidden_dims, num_lstm_layers, fc_dims, out_dims):
        super(VideoEncoder, self).__init__()
        # cnn = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1) # load pre-trained weights
        cnn = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V1)
        modules = list(cnn.children())
        fc = modules[-1]
        modules = modules[:-1]
        self.cnn = nn.Sequential(*modules, nn.Flatten(1, -1), nn.Linear(fc.in_features, lstm_input_dims))

        self.lstm = nn.LSTM(input_size=lstm_input_dims, hidden_size=hidden_dims, num_layers=num_lstm_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, out_dims),
            nn.LayerNorm(out_dims)
        )

    def forward(self, video):
        batch_size = video.shape[0]
        n_frames = video.shape[1]

        cnn_out = torch.unflatten(self.cnn(torch.flatten(video, start_dim=0, end_dim=1)), 0, (batch_size, n_frames))
        video_features, (h_n, c_n) = self.lstm(cnn_out)
        return self.fc(video_features[:, -1, :])

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dims, hidden_dims, num_lstm_layers, fc_dims, out_dims):
        super(TextEncoder, self).__init__()

        self.lstm = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim=embedding_dims),
            nn.LSTM(input_size=embedding_dims, hidden_size=hidden_dims, num_layers=num_lstm_layers, batch_first=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, out_dims),
            nn.LayerNorm(out_dims)
        )

    def forward(self, text):
        text_features, (h_n, c_n) = self.lstm(text)
        return self.fc(text_features[:, -1, :])

class VideoTextMatch(nn.Module):
    def __init__(self, vocab_size, embedding_dims, lstm_input_dims, hidden_dims, text_lstm_layers, video_lstm_layers, fc_dims, out_dims):
        super(VideoTextMatch, self).__init__()

        self.text_encoder = TextEncoder(vocab_size, embedding_dims, hidden_dims, text_lstm_layers, fc_dims, out_dims)
        self.video_encoder = VideoEncoder(lstm_input_dims, hidden_dims, video_lstm_layers, fc_dims, out_dims)
        
    def forward(self, video, text):
        return self.video_encoder(video), self.text_encoder(text)

if __name__ == '__main__':
    vatex = Vatex(is_train=True)
    loader = DataLoader(vatex, batch_size=6, collate_fn=vatex.collate_fn)
    model = VideoTextMatch(vatex.vocab_size(), 100, 100, 32).to(DEVICE)
    video, text, _, __ = next(iter(loader))
    video = video.to(DEVICE)
    text = text.to(DEVICE)
    print(video.shape, text.shape)
    out = model(video, text)
    print(out)