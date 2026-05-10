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
# NEW: ResNet-18 1D + CROSS-ATTENTION FUSION ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
# Key improvements:
#   1. ResNet-18 adapted for 1D ECG: 8 residual blocks, 512-dim output
#      (much deeper than baseline's 3-layer CNN)
#   2. Cross-attention: clinical features ACT AS QUERIES attending to
#      ECG temporal features (keys/values). This lets the model selectively
#      focus on ECG regions relevant to clinical context (e.g., ST-segments
#      when troponin is elevated).
#   3. Combined output: cross-attention result + global ECG pool + clinical
# ══════════════════════════════════════════════════════════════════════════════


class BasicBlock1d(nn.Module):
    """Standard ResNet basic block adapted for 1D signals."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.downsample = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1d18Backbone(nn.Module):
    """ResNet-18 adapted for 12-lead ECG (1D). Produces temporal feature maps."""

    def __init__(self, in_channels: int = 12, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        # 4 stages, 2 blocks each (= 8 residual blocks = ResNet-18)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, dropout=dropout * 0.25)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, dropout=dropout * 0.25)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, dropout=dropout * 0.5)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, dropout=dropout * 0.5)
        self.out_dim = 512

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride, dropout):
        layers = [BasicBlock1d(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Returns temporal feature map (B, 512, T')."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CrossAttentionFusion(nn.Module):
    """Clinical features (query) attend to ECG temporal features (key/value).

    This allows the model to selectively focus on ECG regions most relevant
    to the patient's clinical context. For example, if troponin is elevated,
    the model can attend more to ST-segment regions.
    """

    def __init__(self, ecg_dim: int = 512, clinical_dim: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, ecg_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ecg_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.norm = nn.LayerNorm(ecg_dim)

    def forward(self, ecg_seq, clinical_embed):
        """
        ecg_seq:        (B, T, ecg_dim)  — temporal ECG features
        clinical_embed: (B, clinical_dim) — clinical feature vector
        Returns:        (B, ecg_dim) — cross-attended output
        """
        # Clinical as query (1 token per sample)
        query = self.clinical_proj(clinical_embed).unsqueeze(0)  # (1, B, ecg_dim)
        key = ecg_seq.transpose(0, 1)    # (T, B, ecg_dim)
        value = key

        attn_out, _ = self.cross_attn(query, key, value)  # (1, B, ecg_dim)
        return self.norm(attn_out.squeeze(0))  # (B, ecg_dim)


class CrossAttentionEarlyFusionModel(nn.Module):
    """
    ResNet-18 1D backbone + Cross-Attention fusion.

    Architecture:
      1. Clinical MLP: tabular features -> 128-dim embedding
      2. ResNet-18 1D: 12-lead ECG -> (B, 512, T) temporal features
      3. Cross-attention: clinical embedding queries ECG temporal features
      4. Classifier: [cross_attn_out || global_ecg_pool || clinical_embed] -> logit
    """

    def __init__(self, n_leads: int, ecg_length: int, n_clinical: int,
                 dropout: float = 0.3):
        super().__init__()

        # Clinical encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_clinical, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # ECG backbone
        self.ecg_backbone = ResNet1d18Backbone(
            in_channels=n_leads,
            dropout=dropout,
        )
        ecg_dim = self.ecg_backbone.out_dim  # 512

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            ecg_dim=ecg_dim,
            clinical_dim=128,
            n_heads=4,
            dropout=dropout,
        )

        # Global ECG pooling (complement to cross-attention)
        self.ecg_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier: cross_attn(512) + global_pool(512) + clinical(128) = 1152
        self.classifier = nn.Sequential(
            nn.Linear(ecg_dim + ecg_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, ecg, clinical):
        # Clinical embedding
        clinical_embed = self.clinical_encoder(clinical)   # (B, 128)

        # ECG temporal features
        ecg_features = self.ecg_backbone(ecg)              # (B, 512, T)
        ecg_seq = ecg_features.transpose(1, 2)             # (B, T, 512)

        # Cross-attention: clinical queries ECG
        cross_out = self.cross_attention(ecg_seq, clinical_embed)  # (B, 512)

        # Global ECG pool
        ecg_global = self.ecg_pool(ecg_features).squeeze(-1)      # (B, 512)

        # Fuse all three representations
        fused = torch.cat([cross_out, ecg_global, clinical_embed], dim=1)  # (B, 1152)
        return self.classifier(fused).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# NEW: CLINICAL-CONDITIONED ECG ATTENTION (Phase 2B)
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalOnlyModel(nn.Module):
    def __init__(self, n_clinical: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, ecg, clinical):
        return self.mlp(clinical).squeeze(-1)

def build_model(
    arch_name: str,
    n_leads: int,
    ecg_length: int,
    n_clinical: int,
    dropout: float = 0.3,
):
    if arch_name == "clinical_only":
        return ClinicalOnlyModel(n_clinical=n_clinical, dropout=dropout)
    return EarlyFusionModel()

class ClinicalConditionedFusionModel(nn.Module):
    def __init__(self, n_leads: int, ecg_length: int, n_clinical: int, dropout: float):
        super().__init__()
        
        # 1. Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_clinical, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # 2. ECG Backbone
        # 1x1 Spatial Bottleneck (Learn linear combinations of leads)
        self.spatial_bottleneck = nn.Conv1d(n_leads, n_leads, kernel_size=1)
        
        self.ecg_backbone = nn.Sequential(
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
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # 3. Clinical-Conditioned Multi-Head Attention
        self.ecg_ln = nn.LayerNorm(128)
        self.clin_ln = nn.LayerNorm(128)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, ecg, clinical):
        # Clinical Embedding: (B, 128)
        c_emb = self.clinical_encoder(clinical)
        
        # ECG Spatial & Temporal Features
        e_spatial = self.spatial_bottleneck(ecg)        # (B, 12, T)
        e_feat = self.ecg_backbone(e_spatial)           # (B, 128, T')
        e_seq = e_feat.transpose(1, 2)                  # (B, T', 128)
        e_seq, _ = self.bilstm(e_seq)                   # (B, T', 128)
        
        # Stabilization
        e_seq_norm = self.ecg_ln(e_seq)                 # (B, T', 128)
        c_emb_norm = self.clin_ln(c_emb).unsqueeze(1)   # (B, 1, 128)
        
        # Multi-Head Cross-Attention (Query=Clinical, Key/Value=ECG)
        # attn_out: (B, 1, 128)
        attn_out, attn_weights = self.cross_attn(
            query=c_emb_norm,
            key=e_seq_norm,
            value=e_seq_norm,
            need_weights=True,
            average_attn_weights=False  # Keep all 4 heads
        )
        self.last_attn_weights = attn_weights  # (B, 4, 1, T')
        
        # Residual Pooling (Attention Pool + Global Mean Pool)
        global_mean = e_seq.mean(dim=1)                 # (B, 128)
        pooled_ecg = attn_out.squeeze(1) + global_mean  # (B, 128)
        
        # Fusion
        fused = torch.cat([pooled_ecg, c_emb], dim=1)   # (B, 256)
        self.last_fused_embedding = fused
        return self.classifier(fused).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: SIAMESE DELTA ECG ARCHITECTURE (PHASE 4)
# ══════════════════════════════════════════════════════════════════════════════

class SiameseDeltaFusionModel(nn.Module):
    """
    Explicitly models physiological deviation (E_admit - E_base) while retaining
    the absolute morphological state (E_admit) and clinical context.
    """
    def __init__(self, n_leads=12, ecg_length=5000, n_clinical=81, dropout=0.3):
        super().__init__()
        
        # 1. Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        
        # 2. Shared Siamese ECG Backbone
        self.spatial_bottleneck = nn.Conv1d(n_leads, n_leads, kernel_size=1)
        self.ecg_backbone = nn.Sequential(
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
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # 3. Missing Baseline Token
        self.baseline_missing_embedding = nn.Parameter(torch.zeros(1, 128))
        # Initialize with small normal noise so it's not strictly zero
        nn.init.normal_(self.baseline_missing_embedding, std=0.02)
        
        # 4. Fusion Classifier
        # Concatenated features: [E_admit (128), E_diff (128), E_abs (128), C_emb (128)] = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Prevent overfitting the massive 512-dim temporal space
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        self.last_fused_embedding = None

    def _encode_ecg(self, ecg):
        """Pass single ECG tensor through shared Siamese weights."""
        e_spatial = self.spatial_bottleneck(ecg)
        e_feat = self.ecg_backbone(e_spatial)
        e_seq = e_feat.transpose(1, 2)
        e_seq, _ = self.bilstm(e_seq)
        
        # Max-pool over time
        pooled_ecg, _ = torch.max(e_seq, dim=1)
        return pooled_ecg

    def forward(self, ecg_admit, clinical, ecg_base=None, has_baseline=None):
        B = ecg_admit.size(0)
        
        # 1. Embed Clinical Data
        c_emb = self.clinical_encoder(clinical)  # (B, 128)
        
        # 2. Encode Admission ECG
        e_admit = self._encode_ecg(ecg_admit)    # (B, 128)
        
        # 3. Encode Baseline ECG (Handling missing ones natively)
        if ecg_base is not None and has_baseline is not None:
            e_base_raw = self._encode_ecg(ecg_base) # (B, 128)
            
            # Use gating to replace missing baselines with the learnable token
            mask = has_baseline.view(-1, 1) # (B, 1)
            e_base = mask * e_base_raw + (1.0 - mask) * self.baseline_missing_embedding.expand(B, -1)
        else:
            # Fallback for inference/datasets that haven't been updated yet
            e_base = self.baseline_missing_embedding.expand(B, -1)
            
        # 4. Compute physiological deviation
        e_diff = e_admit - e_base
        e_abs = torch.abs(e_admit - e_base)
        
        # 5. Multimodal Temporal Fusion
        fused = torch.cat([e_admit, e_diff, e_abs, c_emb], dim=1) # (B, 512)
        self.last_fused_embedding = fused
        
        logits = self.classifier(fused).squeeze(-1)
        return logits


class CrossAttentionSiameseModel(nn.Module):
    """
    Explicitly aligns baseline and admission ECGs using Cross-Attention before computing
    delta deviations. Uses Attentive Pooling over the aligned sequence.
    """
    def __init__(self, n_leads=12, ecg_length=5000, n_clinical=81, dropout=0.3):
        super().__init__()
        
        # 1. Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        
        # 2. Shared Siamese ECG Backbone
        self.spatial_bottleneck = nn.Conv1d(n_leads, n_leads, kernel_size=1)
        self.ecg_backbone = nn.Sequential(
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
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # 3. Missing Baseline Token (Sequence level)
        self.baseline_missing_seq = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.baseline_missing_seq, std=0.02)
        
        # 4. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=dropout)
        
        # 5. Attentive Pooling
        # Fusion is [admit_seq (128), e_diff_seq (128), e_abs_seq (128)] = 384
        self.attn_pool = nn.Sequential(
            nn.Linear(384, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 6. Classifier
        # Pooled features (384) + clinical (128) = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Regularize the massive fusion space
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        self.last_fused_embedding = None
        self.last_attn_weights = None

    def _encode_ecg_seq(self, ecg):
        e_spatial = self.spatial_bottleneck(ecg)
        e_feat = self.ecg_backbone(e_spatial)
        e_seq = e_feat.transpose(1, 2)
        e_seq, _ = self.bilstm(e_seq)
        return e_seq

    def forward(self, ecg_admit, clinical, ecg_base=None, has_baseline=None):
        B = ecg_admit.size(0)
        c_emb = self.clinical_encoder(clinical)
        
        admit_seq = self._encode_ecg_seq(ecg_admit) # (B, T, 128)
        T = admit_seq.size(1)
        
        if ecg_base is not None and has_baseline is not None:
            base_seq_raw = self._encode_ecg_seq(ecg_base) # (B, T, 128)
            mask = has_baseline.view(-1, 1, 1)
            base_seq = mask * base_seq_raw + (1.0 - mask) * self.baseline_missing_seq.expand(B, T, -1)
        else:
            base_seq = self.baseline_missing_seq.expand(B, T, -1)
            
        # Cross-Attention: Admit queries Baseline
        aligned_base, _ = self.cross_attn(query=admit_seq, key=base_seq, value=base_seq)
        
        e_diff_seq = admit_seq - aligned_base
        e_abs_seq = torch.abs(admit_seq - aligned_base)
        
        fused_seq = torch.cat([admit_seq, e_diff_seq, e_abs_seq], dim=-1) # (B, T, 384)
        
        # Attentive pooling
        attn_weights = self.attn_pool(fused_seq) # (B, T, 1)
        self.last_attn_weights = attn_weights.transpose(1, 2).unsqueeze(1) # shape: (B, 1, 1, T) for entropy loss
        pooled_fused = (fused_seq * attn_weights).sum(dim=1) # (B, 384)
        
        fused = torch.cat([pooled_fused, c_emb], dim=1) # (B, 512)
        self.last_fused_embedding = fused
        
        logits = self.classifier(fused).squeeze(-1)
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

        # ── CrossAttention architecture: ResNet-18 + cross-attention ────
        if extractor_arch == "crossattn":
            self._crossattn = CrossAttentionEarlyFusionModel(
                n_leads=n_leads,
                ecg_length=ecg_length,
                n_clinical=n_clinical,
                dropout=dropout,
            )
            return
            
        # ── Clinical-Only MLP architecture ──────────────────────────────
        if extractor_arch == "clinical_only":
            self._clinical_only = ClinicalOnlyModel(n_clinical=n_clinical, dropout=dropout)
            return

        # ── Clinical-Gated Attention architecture ───────────────────────
        if extractor_arch == "clinical_gated":
            self._clinical_gated = ClinicalConditionedFusionModel(
                n_leads=n_leads,
                ecg_length=ecg_length,
                n_clinical=n_clinical,
                dropout=dropout,
            )
            return

        # ── Siamese Delta architecture ──────────────────────────────────────
        if extractor_arch == "siamese_delta":
            self._siamese_delta = SiameseDeltaFusionModel(
                n_leads=n_leads,
                ecg_length=ecg_length,
                n_clinical=n_clinical,
                dropout=dropout,
            )
            return

        # ── Siamese Cross-Attention architecture ────────────────────────────
        if extractor_arch == "siamese_crossattn":
            self._siamese_crossattn = CrossAttentionSiameseModel(
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
        if n_clinical == 0:
            self.clinical_projector = None
            clinical_channels = 0
        else:
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

    def forward(self, ecg, clinical, ecg_base=None, has_baseline=None):
        # ── FiLM architecture ───────────────────────────────────────────
        if self.arch == "filmed":
            return self._filmed(ecg, clinical)

        # ── CrossAttention architecture ─────────────────────────────────
        if self.arch == "crossattn":
            return self._crossattn(ecg, clinical)

        # ── Clinical-Only architecture ──────────────────────────────────
        if self.arch == "clinical_only":
            return self._clinical_only(ecg, clinical)

        # ── Clinical-Gated architecture ─────────────────────────────────
        if self.arch == "clinical_gated":
            return self._clinical_gated(ecg, clinical)

        # ── Siamese Delta architecture ──────────────────────────────────
        if self.arch == "siamese_delta":
            return self._siamese_delta(ecg, clinical, ecg_base, has_baseline)

        # ── Siamese Cross-Attention architecture ─────────────────────────
        if self.arch == "siamese_crossattn":
            return self._siamese_crossattn(ecg, clinical, ecg_base, has_baseline)


        # ── Legacy architectures ────────────────────────────────────────
        if self.clinical_projector is None:
            fused_input = ecg
        else:
            clinical_context = self.clinical_projector(clinical)
            clinical_context = clinical_context.unsqueeze(-1).expand(-1, -1, self.ecg_length)
            fused_input = torch.cat([ecg, clinical_context], dim=1)
            
        fused_features = self.shared_extractor(fused_input)
        logits = self.classifier(fused_features).squeeze(-1)
        return logits
