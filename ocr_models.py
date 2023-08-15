# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
import torch.nn as nn
import torch.nn.functional as F

# +
from PIL import ImageOps

class ResizeAndPadHorizontal:
    def __init__(self, target_w, target_h):
        self.target_w = target_w
        self.target_h = target_h

    def __call__(self, img):
        # Calculate scaling factor
        h = img.height
        w = img.width
        scale_factor = self.target_h / h
        
        # Resize image
        new_h = round(h * scale_factor)
        new_w = round(w * scale_factor)        
        
        # Pad image if necessary
        if new_w > self.target_w:
            img = img.resize((self.target_w, new_h))
        elif new_w <= self.target_w:
            img = img.resize((new_w, new_h))
            pad_width = self.target_w - new_w
            img = ImageOps.expand(img, (pad_width, 0, 0, 0), fill=255)

        assert img.height == self.target_h, f'Expected new height ({h}x{scale_factor}={new_h}) to be equal to target {self.target_h} but got {img.size[0]}'
        assert img.width == self.target_w, f'Expected new width to be equal to target {self.target_w} but got {img.size[1]}'
        return img


# -

class CTCCRNNNoStretchV2(nn.Module):
    def __init__(self, image_height, image_width,
                 model_output_len, dropout_rate, conv_channels,
                 lstm_sizes, ids_to_chars):
        super().__init__()
        self.ids_to_chars = ids_to_chars
        num_classes = len(self.ids_to_chars)
        self.model_output_len = model_output_len
        self.image_height = image_height
        self.image_width = image_width
        self.conv_channels = conv_channels

        # CNN part, downsampling 2 times
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=conv_channels[0],
                               kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=conv_channels[0],
                               out_channels=conv_channels[1],
                               kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # first dense
        self.fc1 = nn.Linear(in_features=conv_channels[1] * (self.image_height//4),
                             out_features=conv_channels[1])
        self.dropout = nn.Dropout(dropout_rate)

        # RNN part
        self.lstm1 = nn.LSTM(input_size=conv_channels[1], hidden_size=lstm_sizes[0],
                             bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_sizes[0]*2, hidden_size=lstm_sizes[1],
                             bidirectional=True, batch_first=True)

        self.fc2 = nn.Linear(in_features=2*lstm_sizes[1], out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool2(x)
        
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0],
                                          x.shape[3],
                                          x.shape[1] * x.shape[2])

        x = self.fc1(x)
        x = self.dropout(x)
    
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc2(x)
        x = x.log_softmax(2)
        return x
