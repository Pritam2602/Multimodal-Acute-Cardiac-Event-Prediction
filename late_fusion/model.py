import torch
import torch.nn as nn

from .config import ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE


class ECGBranch(nn.Module):
    """ECG-only encoder that produces a 128-d feature vector."""

    def __init__(self, n_leads: int, ecg_length: int, dropout: float):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, ecg_length)
            conv_out = self._forward_conv(dummy)
            lstm_input_size = conv_out.shape[1]

        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _forward_conv(self, ecg):
        x = self.conv1(ecg)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, ecg):
        x = self._forward_conv(ecg)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.encoder_head(x)


class ClinicalBranch(nn.Module):
    """Clinical-only MLP encoder that produces a 32-d feature vector."""

    def __init__(self, n_clinical: int, dropout: float):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, clinical):
        return self.encoder(clinical)


class ECGOnlyModel(nn.Module):
    """ECG-only baseline using the late-fusion ECG branch."""

    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.ecg_branch = ECGBranch(
            n_leads=n_leads,
            ecg_length=ecg_length,
            dropout=dropout,
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, ecg, clinical=None):
        ecg_feat = self.ecg_branch(ecg)
        return self.classifier(ecg_feat).squeeze(-1)


class ClinicalOnlyModel(nn.Module):
    """Clinical-only baseline using the late-fusion clinical branch."""

    def __init__(
        self,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.clinical_branch = ClinicalBranch(
            n_clinical=n_clinical,
            dropout=dropout,
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, ecg, clinical):
        clinical_feat = self.clinical_branch(clinical)
        return self.classifier(clinical_feat).squeeze(-1)


class LateFusionModel(nn.Module):
    """
    Late-fusion AMI model.

    ECG and clinical inputs are processed independently into modality-specific
    feature vectors. The final prediction is made only after feature fusion.
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
        self.ecg_head = nn.Linear(128, 1)
        self.clinical_head = nn.Linear(32, 1)
        self.fusion_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, ecg, clinical):
        ecg_feat = self.ecg_branch(ecg)
        clinical_feat = self.clinical_branch(clinical)
        ecg_logit = self.ecg_head(ecg_feat)
        clinical_logit = self.clinical_head(clinical_feat)
        combined_logits = torch.cat([ecg_logit, clinical_logit], dim=1)
        final_logit = self.fusion_head(combined_logits)
        return (
            final_logit.squeeze(-1),
            ecg_logit.squeeze(-1),
            clinical_logit.squeeze(-1),
        )
