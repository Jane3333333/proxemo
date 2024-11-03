#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary
from transformers import ViTModel


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, hidden_dim)


class VSGCNN_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_classes, transformer_dim, num_heads, num_layers, dropout=0.2):
        """Constructor

        Args:
            n_classes (int): Num output classes
            in_channels ([type]): Num input channels
            num_groups ([type]): Number of View angle groups
            transformer_dim ([type])
            num_heads ([type])
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            layer_channels (list, optional): Number of channels at each stage.
                                             Defaults to [32, 64, 16].
        """
        super(VSGCNN_Transformer, self).__init__()
        # CNN feature extractor for each image
        self.cnn = CNNFeatureExtractor(in_channels, hidden_dim)

        # Transformer encoder for sequence of features
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim, nhead=num_heads, dim_feedforward=4 * transformer_dim, dropout=dropout
            ),
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, transformer_dim)
        self.classifier = nn.Linear(transformer_dim, n_classes)

    def forward(self, x):
        # x is expected to be of shape (batch_size, num_images, in_channels, height, width)
        batch_size, num_images, _, _, _ = x.size()

        # Apply CNN to each image
        features = []
        for i in range(num_images):
            img_features = self.cnn(x[:, i, :, :, :])  # (batch_size, hidden_dim)
            features.append(img_features)

        # Stack features to form a sequence (batch_size, num_images, hidden_dim)
        features = torch.stack(features, dim=1)

        # Project features to transformer input dimension
        features = self.fc(features)

        # Prepare input for transformer (seq_len, batch_size, transformer_dim)
        features = features.permute(1, 0, 2)

        # Apply transformer encoder
        transformer_output = self.transformer(features)

        # Use the output from the last transformer token for classification
        final_representation = transformer_output[-1]  # (batch_size, transformer_dim)
        output = self.classifier(final_representation)

        return output


# 示例代码
if __name__ == "__main__":
    model = VSGCNN_Transformer(in_channels=3, hidden_dim=256, n_classes=4, transformer_dim=128, num_heads=8, num_layers=4, dropout=0.2)
    image = torch.rand(1, 4, 3, 224, 224)
    # print(summary.summary(model, input_size=(4, 3, 224, 224)))
    a = model(image)
    # print(sum([param.nelement() for param in model.parameters()]))
    print(a.data)

