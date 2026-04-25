import torch
import torch.nn as nn

from .config import ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE


class EarlyFusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, ecg_length: int, dropout: float):
        super().__init__()

        # ── Block 1: wide kernel to capture gross morphology ────────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # ── Block 2: medium kernel ──────────────────────────────────────
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # ── Block 3: with residual skip connection ──────────────────────
        self.conv3_main = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3_skip = nn.Conv1d(64, 128, kernel_size=1)  # channel projection for residual
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ── Block 4: deeper feature refinement ──────────────────────────
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Compute BiLSTM input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, ecg_length)
            d = self.conv1(dummy)
            d = self.conv2(d)
            d = self.pool3(self.conv3_main(d) + self.conv3_skip(d))
            d = self.conv4(d)
            lstm_input_size = d.shape[1]  # 128

        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # ── Temporal attention: learn which timesteps matter ────────────
        # BiLSTM output is 128 (64 * 2 directions)
        self.attn_score = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # CNN feature extraction with residual connection
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool3(self.conv3_main(x) + self.conv3_skip(x))   # residual
        x = self.conv4(x)

        # BiLSTM over temporal dimension
        x = x.transpose(1, 2)               # (B, T, 128)
        x, _ = self.bilstm(x)               # (B, T, 128)

        # Attention-weighted pooling over all timesteps
        scores = self.attn_score(x)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # (B, T, 1)
        x = (x * weights).sum(dim=1)        # (B, 128)

        return self.head(x)


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
        clinical_channels: int = 16,
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
