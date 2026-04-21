import torch
import torch.nn as nn

from .config import ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE


class EarlyFusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, ecg_length: int, dropout: float):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, ecg_length)
            conv_out = self.conv_block(dummy)
            lstm_input_size = conv_out.shape[1]

        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.head(x)


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
        clinical_channels: int = 8,
    ):
        super().__init__()

        self.ecg_length = ecg_length
        self.clinical_channels = clinical_channels

        # Project tabular clinical features into a small set of channels,
        # then broadcast them across the ECG timeline so fusion happens
        # before the shared temporal feature extractor.
        self.clinical_projector = nn.Sequential(
            nn.Linear(n_clinical, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, clinical_channels),
            nn.ReLU(inplace=True),
        )

        self.shared_extractor = EarlyFusionFeatureExtractor(
            in_channels=n_leads + clinical_channels,
            ecg_length=ecg_length,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, ecg, clinical):
        clinical_context = self.clinical_projector(clinical)
        clinical_context = clinical_context.unsqueeze(-1).expand(-1, -1, self.ecg_length)

        fused_input = torch.cat([ecg, clinical_context], dim=1)
        fused_features = self.shared_extractor(fused_input)
        logits = self.classifier(fused_features).squeeze(-1)
        return logits
