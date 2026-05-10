import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model import EarlyFusionFeatureExtractor
from .config import ECG_LEADS, ECG_LENGTH, DROPOUT_RATE, NUM_CLINICAL_FEATURES

class TemporalAttention(nn.Module):
    """
    Computes a weighted average of GRU sequence outputs, allowing the model
    to focus on the most diagnostically relevant phase of the admission.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_outputs, seq_masks):
        # gru_outputs: (B, T, hidden_dim)
        # seq_masks: (B, T) where 1.0 is valid, 0.0 is padded
        
        scores = self.attention(gru_outputs).squeeze(-1) # (B, T)
        
        # Apply mask: set padded timesteps to very large negative value
        scores = scores.masked_fill(seq_masks == 0.0, -1e4)
        
        # Softmax over time
        attn_weights = F.softmax(scores, dim=1) # (B, T)
        
        # Weighted sum of gru outputs
        # (B, 1, T) @ (B, T, hidden_dim) -> (B, 1, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), gru_outputs).squeeze(1) # (B, hidden_dim)
        
        return context, attn_weights


class TemporalMultimodalGRU(nn.Module):
    def __init__(
        self,
        n_leads: int = ECG_LEADS,
        ecg_length: int = ECG_LENGTH,
        n_static_clinical: int = NUM_CLINICAL_FEATURES,
        n_dynamic_clinical: int = 3, # trop_t, trop_time_delta_t, ecg_time_delta_t
        dropout: float = DROPOUT_RATE
    ):
        super().__init__()
        
        # ── 1. Timestep Encoders ─────────────────────────────────────────────
        from .lead_aware_model import LeadAwareHybridExtractor, AnatomicalRegionalAttention
        self.ecg_encoder = LeadAwareHybridExtractor(
            in_channels=n_leads,
            ecg_length=ecg_length,
            dropout=dropout
        ) # Returns local_feats (12, 64) and global_context (128)
        
        # Anatomical Regional Attention: "Attend to regions that changed the most"
        self.lead_delta_attention = AnatomicalRegionalAttention(feature_dim=64)
        
        self.ecg_projector = nn.Sequential(
            nn.Linear(64 + 64 + 128, 128), # local_context + delta_context + global_context
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_static_clinical + n_dynamic_clinical, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        ) # Outputs 64D
        
        fusion_dim = 128 + 64 # 192
        
        # ── 2. Temporal Trajectory Engine ────────────────────────────────────
        self.gru_hidden_size = 128
        self.gru = nn.GRU(
            input_size=fusion_dim * 2, # Doubled for Cross-Timestep Delta Embeddings
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        ) # Outputs 128D
        
        self.temporal_attention = TemporalAttention(hidden_dim=self.gru_hidden_size)
        
        # ── 3. Final Progression Classifier ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.last_attn_weights = None
        self.last_trajectory_embedding = None

    def forward(self, ecg_seq, trop_seq, trop_times, ecg_times, seq_masks, clinical_static):
        """
        ecg_seq: (B, T, 12, 5000)
        trop_seq: (B, T)
        trop_times: (B, T)
        ecg_times: (B, T)
        seq_masks: (B, T)
        clinical_static: (B, 71)
        """
        B, T, C, L = ecg_seq.shape
        
        # 1. Prepare dynamic inputs
        # We need to process timestep by timestep, but batching across time is much faster.
        # Flatten B and T: (B*T, ...)
        
        ecg_flat = ecg_seq.view(B * T, C, L)
        
        trop_flat = trop_seq.contiguous().view(B * T, 1)
        trop_times_flat = trop_times.contiguous().view(B * T, 1)
        ecg_times_flat = ecg_times.contiguous().view(B * T, 1)
        
        clinical_static_expanded = clinical_static.unsqueeze(1).expand(B, T, -1).contiguous().view(B * T, -1)
        
        clinical_flat = torch.cat([clinical_static_expanded, trop_flat, trop_times_flat, ecg_times_flat], dim=1)
        
        # 2. Extract Timestep Embeddings
        local_feats_flat, global_context_flat = self.ecg_encoder(ecg_flat) # (B*T, 12, 64) and (B*T, 128)
        
        local_feats = local_feats_flat.view(B, T, 12, 64)
        global_context = global_context_flat.view(B, T, 128)
        
        # ── Compute Cross-Timestep Lead Deltas ──
        # By taking the morphological difference of each lead BEFORE attention,
        # we can explicitly ignore chronic abnormalities (which have delta=0)
        lead_deltas = torch.zeros_like(local_feats)
        lead_deltas[:, 1:, :, :] = local_feats[:, 1:, :, :] - local_feats[:, :-1, :, :]
        
        # Flatten back for attention processing
        lead_deltas_flat = lead_deltas.view(B * T, 12, 64)
        
        # Compute spatial attention STRICTLY based on morphological change
        delta_context_flat, spatial_weights = self.lead_delta_attention(lead_deltas_flat) # (B*T, 64), (B*T, 12)
        
        # Cache for interpretability
        self.ecg_encoder.last_spatial_weights = spatial_weights
        
        # Apply the delta-derived weights to the absolute embeddings
        local_context_flat = torch.sum(spatial_weights.unsqueeze(-1) * local_feats_flat, dim=1) # (B*T, 64)
        
        # Fuse Local (Absolute), Local (Delta), and Global into a final ECG embedding
        ecg_fused_flat = torch.cat([local_context_flat, delta_context_flat, global_context_flat], dim=-1) # 256D
        ecg_emb_flat = self.ecg_projector(ecg_fused_flat) # (B*T, 128)
        
        clinical_emb_flat = self.clinical_encoder(clinical_flat) # (B*T, 64)
        
        # 3. Timestep Fusion
        fusion_flat = torch.cat([ecg_emb_flat, clinical_emb_flat], dim=1) # (B*T, 192)
        fusion_seq = fusion_flat.view(B, T, -1) # (B, T, 192)
        
        # 3b. Cross-Timestep Delta Embeddings
        # delta_t = emb_t - emb_{t-1}
        delta_seq = torch.zeros_like(fusion_seq)
        delta_seq[:, 1:, :] = fusion_seq[:, 1:, :] - fusion_seq[:, :-1, :]
        
        # Concatenate absolute embeddings with their temporal deltas
        fusion_seq_with_deltas = torch.cat([fusion_seq, delta_seq], dim=-1) # (B, T, 384)
        
        # 4. Temporal Sequence Modeling (GRU)
        # We must pack the sequence to avoid training the GRU on pure padding
        seq_lengths = seq_masks.sum(dim=1).cpu().int()
        
        # Ensure minimum length of 1 for packing (though our masks are guaranteed >=1)
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        # pack_padded_sequence requires sequences to be sorted by length in descending order
        # unless enforce_sorted=False (PyTorch 1.1+)
        packed_input = pack_padded_sequence(fusion_seq_with_deltas, seq_lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, _ = self.gru(packed_input)
        
        gru_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)
        # gru_output shape: (B, T, 256)
        
        # 5. Temporal Attention
        context, attn_weights = self.temporal_attention(gru_output, seq_masks)
        
        # Save for interpretability and contrastive loss
        self.last_attn_weights = attn_weights
        self.last_trajectory_embedding = context
        
        # 6. Classification
        logits = self.classifier(context)
        
        return logits
