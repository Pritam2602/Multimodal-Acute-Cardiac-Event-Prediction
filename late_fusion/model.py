import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ECG_BRANCH_TYPE,
    ECG_CNN_FILTERS,
    ECG_LENGTH,
    ECG_LEADS,
    ECG_LSTM_HIDDEN_SIZE,
    FUSION_HIDDEN_DIM,
    FUSION_TYPE,
    NUM_CLINICAL_FEATURES,
    DROPOUT_RATE,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        padding1 = ((7 - 1) * dilation) // 2
        padding2 = ((5 - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=padding1,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=1,
            padding=padding2,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.relu(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(device=x.device, dtype=x.dtype)
        return self.dropout(x)


class TemporalPyramidPooling(nn.Module):
    def __init__(self, bins: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.bins = bins

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence_cf = sequence.transpose(1, 2)
        pooled_features = []
        for bin_size in self.bins:
            pooled = F.adaptive_avg_pool1d(sequence_cf, bin_size)
            pooled_features.append(pooled.flatten(start_dim=1))
        return torch.cat(pooled_features, dim=1)


class ECGBranch(nn.Module):
    """Lead-aware dilated CNN + BiLSTM + Transformer ECG encoder with temporal pyramid pooling."""

    def __init__(
        self,
        n_leads: int,
        ecg_length: int,
        dropout: float,
        cnn_filters: int = ECG_CNN_FILTERS,
        lstm_hidden_size: int = ECG_LSTM_HIDDEN_SIZE,
    ):
        super().__init__()

        self.n_leads = n_leads
        self.ecg_length = ecg_length
        self.cnn_filters = cnn_filters
        self.lstm_hidden_size = lstm_hidden_size
        self.lead_multiplier = max(2, math.ceil(cnn_filters / n_leads))
        lead_temporal_channels = n_leads * self.lead_multiplier

        self.lead_temporal_stem = nn.Sequential(
            nn.Conv1d(
                n_leads,
                lead_temporal_channels,
                kernel_size=15,
                padding=7,
                groups=n_leads,
                bias=False,
            ),
            nn.BatchNorm1d(lead_temporal_channels),
            nn.ReLU(inplace=True),
        )

        self.cross_lead_mixer = nn.Sequential(
            nn.Conv2d(
                self.lead_multiplier,
                cnn_filters,
                kernel_size=(n_leads, 1),
                bias=False,
            ),
            nn.BatchNorm2d(cnn_filters),
            nn.ReLU(inplace=True),
        )

        self.stem_pool = nn.MaxPool1d(kernel_size=4, stride=4)

        stage_channels = [
            cnn_filters,
            cnn_filters * 2,
            cnn_filters * 4,
            cnn_filters * 8,
        ]
        self.res_stage1 = self._make_stage(stage_channels[0], stage_channels[0], num_blocks=2, stride=1, dropout=dropout, dilation=1)
        self.res_stage2 = self._make_stage(stage_channels[0], stage_channels[1], num_blocks=2, stride=2, dropout=dropout, dilation=2)
        self.res_stage3 = self._make_stage(stage_channels[1], stage_channels[2], num_blocks=2, stride=2, dropout=dropout, dilation=4)
        self.res_stage4 = self._make_stage(stage_channels[2], stage_channels[3], num_blocks=2, stride=2, dropout=dropout, dilation=8)

        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, ecg_length)
            conv_out = self._forward_cnn(dummy)
            sequence_input_size = conv_out.shape[1]

        self.bilstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.sequence_dim = lstm_hidden_size * 2
        self.positional_encoding = PositionalEncoding(self.sequence_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_dim,
            nhead=TRANSFORMER_NUM_HEADS,
            dim_feedforward=TRANSFORMER_FF_DIM,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_NUM_LAYERS,
        )
        self.temporal_pyramid_pool = TemporalPyramidPooling(bins=(1, 2, 4))
        self.attention_query = nn.Parameter(torch.randn(1, 1, self.sequence_dim))
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.sequence_dim,
            num_heads=TRANSFORMER_NUM_HEADS,
            dropout=dropout,
            batch_first=True,
        )

        self.embedding_dim = self.sequence_dim
        self.attention = nn.Linear(self.embedding_dim, 1)
        pooled_dim = self.sequence_dim + self.sequence_dim + (self.sequence_dim * sum((1, 2, 4)))
        self.encoder_head = nn.Sequential(
            nn.Linear(pooled_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(self.embedding_dim, 1)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
        dilation: int,
    ):
        blocks = [ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout, dilation=dilation)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout, dilation=dilation))
        return nn.Sequential(*blocks)

    def _forward_cnn(self, ecg):
        x = self.lead_temporal_stem(ecg)
        batch_size, _channels, time_steps = x.shape
        x = x.view(batch_size, self.lead_multiplier, self.n_leads, time_steps)
        x = self.cross_lead_mixer(x).squeeze(2)
        x = self.stem_pool(x)
        x = self.res_stage1(x)
        x = self.res_stage2(x)
        x = self.res_stage3(x)
        x = self.res_stage4(x)
        return x

    def forward_features(self, ecg, return_debug: bool = False):
        x = self._forward_cnn(ecg)
        x = x.transpose(1, 2)
        sequence, _ = self.bilstm(x)
        sequence = self.positional_encoding(sequence)
        sequence = self.transformer_encoder(sequence)

        pooled_query = self.attention_query.expand(sequence.size(0), -1, -1)
        attended_summary, multihead_attention_weights = self.multihead_attention(
            pooled_query,
            sequence,
            sequence,
            need_weights=True,
            average_attn_weights=False,
        )
        attended_summary = attended_summary.squeeze(1)

        pyramid_features = self.temporal_pyramid_pool(sequence)
        attention_logits = self.attention(sequence).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(sequence * attention_weights.unsqueeze(-1), dim=1)
        combined_features = torch.cat([pooled, attended_summary, pyramid_features], dim=1)
        embedding = self.encoder_head(combined_features)

        if return_debug:
            return embedding, {
                "attention_weights": attention_weights,
                "multihead_attention_weights": multihead_attention_weights,
                "sequence_features": sequence,
                "pooled_features": pooled,
                "attended_summary": attended_summary,
                "pyramid_features": pyramid_features,
            }

        return embedding

    def forward(self, ecg, return_debug: bool = False):
        if return_debug:
            embedding, debug = self.forward_features(ecg, return_debug=True)
            logits = self.logit_head(embedding).squeeze(-1)
            debug["embedding"] = embedding
            return logits, debug

        embedding = self.forward_features(ecg, return_debug=False)
        return self.logit_head(embedding).squeeze(-1)


class ClinicalBranch(nn.Module):
    """Clinical-only MLP that produces a branch embedding and logit."""

    def __init__(self, n_clinical: int, dropout: float, embedding_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.logit_head = nn.Linear(embedding_dim, 1)

    def forward_features(self, clinical):
        return self.encoder(clinical)

    def forward(self, clinical):
        embedding = self.forward_features(clinical)
        return self.logit_head(embedding).squeeze(-1)


class LateFusionModel(nn.Module):
    """
    Late-fusion AMI model.

    ECG and clinical inputs are processed independently into embeddings.
    The final prediction is made from the concatenated branch embeddings.
    """

    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
        cnn_filters: int = ECG_CNN_FILTERS,
        lstm_hidden_size: int = ECG_LSTM_HIDDEN_SIZE,
        fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
    ):
        super().__init__()

        self.ecg_branch = ECGBranch(
            n_leads=n_leads,
            ecg_length=ecg_length,
            dropout=dropout,
            cnn_filters=cnn_filters,
            lstm_hidden_size=lstm_hidden_size,
        )
        self.clinical_branch = ClinicalBranch(
            n_clinical=n_clinical,
            dropout=dropout,
            embedding_dim=self.ecg_branch.embedding_dim,
        )

        embedding_dim = self.ecg_branch.embedding_dim
        fused_input_dim = embedding_dim * 2
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

        self.architecture_metadata = {
            "ecg_branch_type": ECG_BRANCH_TYPE,
            "fusion_type": FUSION_TYPE,
            "ecg_length": ecg_length,
            "n_leads": n_leads,
            "n_clinical": n_clinical,
            "dropout": dropout,
            "cnn_filters": cnn_filters,
            "lstm_hidden_size": lstm_hidden_size,
            "fusion_hidden_dim": fusion_hidden_dim,
            "ecg_embedding_dim": self.ecg_branch.embedding_dim,
        }

    def forward_with_branches(self, ecg, clinical):
        ecg_embedding = self.ecg_branch.forward_features(ecg)
        clinical_embedding = self.clinical_branch.forward_features(clinical)
        gate_input = torch.cat([ecg_embedding, clinical_embedding], dim=1)
        gate = torch.sigmoid(self.fusion_gate(gate_input))
        gated_ecg_embedding = gate * ecg_embedding
        fused = torch.cat([gated_ecg_embedding, clinical_embedding], dim=1)
        final_logit = self.fusion_head(fused).squeeze(-1)
        ecg_logit = self.ecg_branch.logit_head(ecg_embedding).squeeze(-1)
        clinical_logit = self.clinical_branch.logit_head(clinical_embedding).squeeze(-1)
        return {
            "fusion_logits": final_logit,
            "ecg_logits": ecg_logit,
            "clinical_logits": clinical_logit,
            "ecg_embedding": ecg_embedding,
            "gated_ecg_embedding": gated_ecg_embedding,
            "clinical_embedding": clinical_embedding,
            "fusion_gate": gate,
        }

    def forward(self, ecg, clinical):
        return self.forward_with_branches(ecg, clinical)["fusion_logits"]


def get_architecture_metadata(model: LateFusionModel) -> dict:
    metadata = dict(getattr(model, "architecture_metadata", {}))
    metadata["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    metadata["trainable_parameter_count"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return metadata
