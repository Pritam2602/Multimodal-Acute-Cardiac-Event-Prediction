import torch
import torch.nn as nn

from .config import ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE


# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL ARCHITECTURES (baseline / resnet) — kept for backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

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


class ResidualECGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class ResNetEarlyFusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, ecg_length: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            ResidualECGBlock(64, 64, stride=2, dropout=dropout * 0.25),
            ResidualECGBlock(64, 128, stride=2, dropout=dropout * 0.25),
            ResidualECGBlock(128, 128, stride=2, dropout=dropout * 0.25),
            ResidualECGBlock(128, 256, stride=2, dropout=dropout * 0.25),
            ResidualECGBlock(256, 256, stride=2, dropout=dropout * 0.25),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(256, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        scores = self.attn_score(x)
        weights = torch.softmax(scores, dim=1)
        x = (x * weights).sum(dim=1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# NEW: FiLM-CONDITIONED MULTI-SCALE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
# Key improvements over baseline:
#   1. Multi-scale convolutions capture both fine (QRS) and broad (ST) features
#   2. FiLM conditioning: clinical features MODULATE ECG features (scale+shift)
#      instead of being broadcast as flat channels that the CNN ignores
#   3. Squeeze-and-Excitation channel attention
#   4. 2-layer BiLSTM for richer temporal modelling
#   5. Dual-path classifier: attention-pooled ECG + direct clinical branch
# ══════════════════════════════════════════════════════════════════════════════


class SqueezeExcitation(nn.Module):
    """Channel attention: learns which feature channels are most important."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.se(x).unsqueeze(-1)      # (B, C, 1)
        return x * w


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: clinical features generate
    per-channel scale (gamma) and shift (beta) for ECG feature maps."""
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, channels)
        self.beta_net = nn.Linear(cond_dim, channels)
        # Initialize gamma near 1 and beta near 0 so FiLM starts as identity
        nn.init.ones_(self.gamma_net.weight.data[:, 0] if cond_dim > 0 else self.gamma_net.weight.data)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, ecg_features, conditioning):
        # ecg_features: (B, C, T),  conditioning: (B, cond_dim)
        gamma = self.gamma_net(conditioning).unsqueeze(-1)  # (B, C, 1)
        beta = self.beta_net(conditioning).unsqueeze(-1)    # (B, C, 1)
        return gamma * ecg_features + beta


class MultiScaleConvBlock(nn.Module):
    """Parallel convolutions at 3 scales + SE channel attention + residual."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        # Split output channels across 3 scales
        ch3 = out_ch // 3
        ch7 = out_ch // 3
        ch15 = out_ch - ch3 - ch7  # remainder goes to wide branch

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, ch3, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(ch3),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, ch7, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(ch7),
            nn.ReLU(inplace=True),
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(in_ch, ch15, kernel_size=15, stride=stride, padding=7, bias=False),
            nn.BatchNorm1d(ch15),
            nn.ReLU(inplace=True),
        )

        self.se = SqueezeExcitation(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual projection
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.branch3(x), self.branch7(x), self.branch15(x)], dim=1)
        out = self.se(out)
        out = self.dropout(out)
        return self.act(out + self.skip(x))


class FiLMedEarlyFusionModel(nn.Module):
    """
    FiLM-conditioned multi-scale early fusion model.

    Architecture:
      1. Clinical encoder → 64-d conditioning vector
      2. Multi-scale CNN stem processes raw 12-lead ECG
      3. 4 residual blocks with FiLM conditioning after each —
         clinical features modulate ECG feature maps via learned scale/shift
      4. 2-layer BiLSTM captures long-range temporal dependencies
      5. Attention-weighted temporal pooling
      6. Dual-path classifier: [ECG features ∥ clinical features] → logit
    """
    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()

        cond_dim = 64  # clinical conditioning dimension

        # ── Clinical encoder (separate pathway) ─────────────────────────
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, cond_dim),
            nn.ReLU(inplace=True),
        )

        # ── Multi-scale CNN stem ────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),      # 5000 → 2500
        )

        # ── FiLM-conditioned residual blocks ────────────────────────────
        self.block1 = MultiScaleConvBlock(64,  64,  stride=2, dropout=dropout * 0.25)  # 2500 → 1250
        self.film1  = FiLMLayer(cond_dim, 64)

        self.block2 = MultiScaleConvBlock(64,  128, stride=2, dropout=dropout * 0.25)  # 1250 → 625
        self.film2  = FiLMLayer(cond_dim, 128)

        self.block3 = MultiScaleConvBlock(128, 256, stride=2, dropout=dropout * 0.25)  # 625 → 313
        self.film3  = FiLMLayer(cond_dim, 256)

        self.block4 = MultiScaleConvBlock(256, 256, stride=2, dropout=dropout * 0.25)  # 313 → 157
        self.film4  = FiLMLayer(cond_dim, 256)

        # ── 2-layer BiLSTM for temporal modelling ───────────────────────
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout * 0.5,
        )
        lstm_out_dim = 256  # 128 * 2 directions

        # ── Temporal attention pooling ──────────────────────────────────
        self.attn_score = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # ── Dual-path classifier ────────────────────────────────────────
        # Concatenate attention-pooled ECG features with clinical features
        classifier_in = lstm_out_dim + cond_dim   # 256 + 64 = 320
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, ecg, clinical):
        # ── Clinical conditioning vector ────────────────────────────────
        cond = self.clinical_encoder(clinical)          # (B, 64)

        # ── ECG pathway with FiLM modulation ────────────────────────────
        x = self.stem(ecg)                              # (B, 64, 2500)

        x = self.block1(x)                              # (B, 64, 1250)
        x = self.film1(x, cond)                         # clinical modulates ECG

        x = self.block2(x)                              # (B, 128, 625)
        x = self.film2(x, cond)

        x = self.block3(x)                              # (B, 256, 313)
        x = self.film3(x, cond)

        x = self.block4(x)                              # (B, 256, 157)
        x = self.film4(x, cond)

        # ── BiLSTM temporal modelling ───────────────────────────────────
        x = x.transpose(1, 2)                           # (B, T, 256)
        x, _ = self.bilstm(x)                           # (B, T, 256)

        # ── Attention-weighted pooling ──────────────────────────────────
        scores = self.attn_score(x)                     # (B, T, 1)
        weights = torch.softmax(scores, dim=1)          # (B, T, 1)
        ecg_features = (x * weights).sum(dim=1)         # (B, 256)

        # ── Dual-path fusion for classification ─────────────────────────
        combined = torch.cat([ecg_features, cond], dim=1)  # (B, 320)
        logits = self.classifier(combined).squeeze(-1)     # (B,)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_clinical: int = NUM_CLINICAL_FEATURES,
        dropout: float = DROPOUT_RATE,
        clinical_channels: int = 16,
        extractor_arch: str = "baseline",
    ):
        super().__init__()

        self.arch = extractor_arch

        # ── FiLM architecture: completely different forward pass ────────
        if extractor_arch == "filmed":
            self._filmed = FiLMedEarlyFusionModel(
                n_leads=n_leads,
                ecg_length=ecg_length,
                n_clinical=n_clinical,
                dropout=dropout,
            )
            return

        # ── Legacy architectures (baseline / resnet) ────────────────────
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

        extractor_cls = {
            "baseline": EarlyFusionFeatureExtractor,
            "resnet": ResNetEarlyFusionFeatureExtractor,
        }.get(extractor_arch)
        if extractor_cls is None:
            raise ValueError(f"Unknown extractor_arch={extractor_arch!r}")

        self.shared_extractor = extractor_cls(
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
        # ── FiLM architecture ───────────────────────────────────────────
        if self.arch == "filmed":
            return self._filmed(ecg, clinical)

        # ── Legacy architectures ────────────────────────────────────────
        clinical_context = self.clinical_projector(clinical)
        clinical_context = clinical_context.unsqueeze(-1).expand(-1, -1, self.ecg_length)

        fused_input = torch.cat([ecg, clinical_context], dim=1)
        fused_features = self.shared_extractor(fused_input)
        logits = self.classifier(fused_features).squeeze(-1)
        return logits
