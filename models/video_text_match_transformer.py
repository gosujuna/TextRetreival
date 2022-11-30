import torch.nn as nn
import torchvision
import torch
from torch.utils.data import DataLoader
from data.vatex import Vatex
from utils.device import DEVICE
import math

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
        x = self.fc(video_features[:, -1, :])
        print(x.shape)
        return self.fc(video_features[:, -1, :])



class TextEncoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_hid, nlayers, dropout):
        super(TextEncoderTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)




                
    def forward(self,text):

        text = self.encoder(text) * math.sqrt(self.d_model)
        text = self.pos_encoder(text)
        output = self.transformer_encoder(text)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, text):

        text = text + self.pe[:text.size(0)]
        return self.dropout(text)      

        


class VideoTextMatch(nn.Module):
    def __init__(self, vocab_size, video_encoder_cfg, text_encoder_cfg, out_dims):
        super(VideoTextMatch, self).__init__()

        self.text_encoder = TextEncoderTransformer(
            vocab_size, 
            text_encoder_cfg['emsize'], 
            text_encoder_cfg['d_hid'], 
            text_encoder_cfg['nlayers'], 
            text_encoder_cfg['nhead'],
            text_encoder_cfg['dropout'] ,

        )
        self.video_encoder = VideoEncoder(
            video_encoder_cfg['lstm_input_dims'], 
            video_encoder_cfg['hidden_dims'], 
            video_encoder_cfg['lstm_layers'], 
            video_encoder_cfg['fc_dims'], 
            out_dims
        )

        
    def forward(self, video, text):
        return self.video_encoder(video), self.text_encoder(text)


def generate_square_subsequent_mask(sz):

    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if __name__ == '__main__':
    vatex = Vatex(is_train=True)
    loader = DataLoader(vatex, batch_size=6, collate_fn=vatex.collate_fn)
    model = VideoTextMatch(vatex.vocab_size(), 100, 100, 32).to(DEVICE)






    #src_mask = generate_square_subsequent_mask(bptt).to(DEVICE)

    video, text, _, __ = next(iter(loader))
    video = video.to(DEVICE)
    text = text.to(DEVICE)
    print(video.shape, text.shape)
    out = model(video, text)
    print(out)