import torch
import torch.nn as nn
import torch.nn.functional as F

from early_fusion.config import ECG_LEADS, ECG_LENGTH, DROPOUT_RATE, NUM_CLINICAL_FEATURES
from early_fusion.temporal_model import TemporalAttention
from early_fusion.lead_aware_model import LeadAwareHybridExtractor, AnatomicalRegionalAttention

class TemporalECGBranch(nn.Module):
    def __init__(self, n_leads=ECG_LEADS, ecg_length=ECG_LENGTH, dropout=DROPOUT_RATE):
        super().__init__()
        
        self.ecg_encoder = LeadAwareHybridExtractor(
            in_channels=n_leads, ecg_length=ecg_length, dropout=dropout
        )
        self.lead_delta_attention = AnatomicalRegionalAttention(feature_dim=64)
        
        self.ecg_projector = nn.Sequential(
            nn.Linear(64 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.gru_hidden_size = 128
        self.gru = nn.GRU(
            input_size=128 * 2, # Doubled for cross-timestep delta embeddings
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.temporal_attention = TemporalAttention(hidden_dim=self.gru_hidden_size)
        
        # Auxiliary classifier to force branch specialization
        self.aux_classifier = nn.Linear(self.gru_hidden_size, 1)

    def forward(self, ecg_seq, seq_masks):
        B, T, C, L = ecg_seq.shape
        ecg_flat = ecg_seq.view(B * T, C, L)
        
        local_feats_flat, global_context_flat = self.ecg_encoder(ecg_flat)
        local_feats = local_feats_flat.view(B, T, 12, 64)
        
        lead_deltas = torch.zeros_like(local_feats)
        lead_deltas[:, 1:, :, :] = local_feats[:, 1:, :, :] - local_feats[:, :-1, :, :]
        
        lead_deltas_flat = lead_deltas.view(B * T, 12, 64)
        delta_context_flat, spatial_weights = self.lead_delta_attention(lead_deltas_flat)
        self.ecg_encoder.last_spatial_weights = spatial_weights
        
        local_context_flat = torch.sum(spatial_weights.unsqueeze(-1) * local_feats_flat, dim=1)
        ecg_fused_flat = torch.cat([local_context_flat, delta_context_flat, global_context_flat], dim=-1)
        
        ecg_emb_flat = self.ecg_projector(ecg_fused_flat)
        ecg_seq_emb = ecg_emb_flat.view(B, T, 128)
        
        delta_seq = torch.zeros_like(ecg_seq_emb)
        delta_seq[:, 1:, :] = ecg_seq_emb[:, 1:, :] - ecg_seq_emb[:, :-1, :]
        
        gru_input = torch.cat([ecg_seq_emb, delta_seq], dim=-1)
        
        lengths = seq_masks.sum(dim=1).cpu().int()
        lengths = torch.clamp(lengths, min=1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, _ = self.gru(packed_input)
        gru_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=T
        )
        
        context, attn_weights = self.temporal_attention(gru_outputs, seq_masks)
        aux_logit = self.aux_classifier(context).squeeze(-1)
        
        return context, attn_weights, aux_logit


class TemporalClinicalBranch(nn.Module):
    def __init__(self, n_static=NUM_CLINICAL_FEATURES, n_dynamic=3, dropout=DROPOUT_RATE):
        super().__init__()
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_static + n_dynamic, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.gru_hidden_size = 64
        self.gru = nn.GRU(
            input_size=64 * 2, # Doubled for cross-timestep deltas
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.temporal_attention = TemporalAttention(hidden_dim=self.gru_hidden_size)
        
        self.aux_classifier = nn.Linear(self.gru_hidden_size, 1)

    def forward(self, clinical_flat, seq_masks, B, T):
        clin_emb_flat = self.clinical_encoder(clinical_flat)
        clin_seq_emb = clin_emb_flat.view(B, T, 64)
        
        delta_seq = torch.zeros_like(clin_seq_emb)
        delta_seq[:, 1:, :] = clin_seq_emb[:, 1:, :] - clin_seq_emb[:, :-1, :]
        
        gru_input = torch.cat([clin_seq_emb, delta_seq], dim=-1)
        
        lengths = seq_masks.sum(dim=1).cpu().int()
        lengths = torch.clamp(lengths, min=1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, _ = self.gru(packed_input)
        gru_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=T
        )
        
        context, attn_weights = self.temporal_attention(gru_outputs, seq_masks)
        aux_logit = self.aux_classifier(context).squeeze(-1)
        
        return context, attn_weights, aux_logit


class TemporalHybridLateFusion(nn.Module):
    """
    Hybrid Late Fusion Temporal Architecture.
    Disentangles the ECG morphological trajectory from the Clinical physiological trajectory.
    """
    def __init__(
        self,
        n_leads=ECG_LEADS,
        ecg_length=ECG_LENGTH,
        n_static_clinical=NUM_CLINICAL_FEATURES,
        n_dynamic_clinical=3,
        dropout=DROPOUT_RATE
    ):
        super().__init__()
        
        self.ecg_branch = TemporalECGBranch(n_leads, ecg_length, dropout)
        self.clinical_branch = TemporalClinicalBranch(n_static_clinical, n_dynamic_clinical, dropout)
        
        # Cross-Modal Gating: Clinical context gates ECG context
        self.clinical_to_ecg_gate = nn.Sequential(
            nn.Linear(self.clinical_branch.gru_hidden_size, self.ecg_branch.gru_hidden_size),
            nn.Sigmoid()
        )
        
        # Final Fusion Classifier
        fusion_dim = self.ecg_branch.gru_hidden_size + self.clinical_branch.gru_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + 2, 64), # +2 for the aux logits
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.last_attn_weights = None 
        self.last_trajectory_embedding = None

    def forward(self, ecg_seq, trop_seq, trop_times, ecg_times, seq_masks, clinical_static):
        B, T, C, L = ecg_seq.shape
        
        # Prepare clinical inputs
        trop_flat = trop_seq.contiguous().view(B * T, 1)
        trop_times_flat = trop_times.contiguous().view(B * T, 1)
        ecg_times_flat = ecg_times.contiguous().view(B * T, 1)
        clinical_static_expanded = clinical_static.unsqueeze(1).expand(B, T, -1).contiguous().view(B * T, -1)
        clinical_flat = torch.cat([clinical_static_expanded, trop_flat, trop_times_flat, ecg_times_flat], dim=1)
        
        # Branch Forward Passes
        ecg_context, ecg_attn, ecg_logit = self.ecg_branch(ecg_seq, seq_masks)
        clin_context, clin_attn, clin_logit = self.clinical_branch(clinical_flat, seq_masks, B, T)
        
        # Cross-Modal Gating
        gate = self.clinical_to_ecg_gate(clin_context)
        gated_ecg_context = ecg_context * gate
        
        # Late Fusion
        fusion_input = torch.cat([gated_ecg_context, clin_context, ecg_logit.unsqueeze(1), clin_logit.unsqueeze(1)], dim=1)
        final_logit = self.classifier(fusion_input).squeeze(-1)
        
        # Save for interpretability / loss
        self.last_attn_weights = ecg_attn # Monitor ECG temporal attention
        self.last_trajectory_embedding = fusion_input 
        
        return final_logit, ecg_logit, clin_logit

    @property
    def ecg_encoder(self):
        # Alias so interpretability tools can access last_spatial_weights easily
        return self.ecg_branch.ecg_encoder
