import torch
import torch.nn as nn

from .config import ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE


class ECGBranch(nn.Module):
    """ECG-only encoder that produces a dense embedding and branch logit."""

    def __init__(self, n_leads: int, ecg_length: int, dropout: float):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
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
            dummy = torch.zeros(1, n_leads, ecg_length)
            conv_out = self.conv_block(dummy)
            lstm_input_size = conv_out.shape[1]

        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_score = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.encoder_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(128, 1)

    def forward(self, ecg):
        x = self.conv_block(ecg)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        weights = torch.softmax(self.attn_score(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        features = self.encoder_head(pooled)
        logit = self.logit_head(features).squeeze(-1)
        return features, logit


class ClinicalBranch(nn.Module):
    """Clinical-only MLP that produces a dense embedding and branch logit."""

    def __init__(self, n_clinical: int, dropout: float):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(64, 1)

    def forward(self, clinical):
        features = self.encoder(clinical)
        logit = self.logit_head(features).squeeze(-1)
        return features, logit


class LateFusionModel(nn.Module):
    """
    Late-fusion AMI model.

    ECG and clinical inputs are processed into learned branch embeddings.
    The final prediction is made from both embeddings plus branch logits.
    """

    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()

        self.ecg_branch = ECGBranch(
            n_leads=n_leads,
            ecg_length=ecg_length,
            dropout=dropout,
        )
        self.clinical_branch = ClinicalBranch(
            n_clinical=n_clinical,
            dropout=dropout,
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 64 + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, ecg, clinical):
        ecg_features, ecg_logit = self.ecg_branch(ecg)
        clinical_features, clinical_logit = self.clinical_branch(clinical)

        fusion_features = torch.cat(
            [
                ecg_features,
                clinical_features,
                ecg_logit.unsqueeze(1),
                clinical_logit.unsqueeze(1),
            ],
            dim=1,
        )
        final_logit = self.fusion_head(fusion_features).squeeze(-1)
        return final_logit
