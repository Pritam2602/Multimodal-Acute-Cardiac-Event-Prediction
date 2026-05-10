import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EarlyFusionFeatureExtractor
from .config import ECG_LEADS, ECG_LENGTH, DROPOUT_RATE

class SpatialLeadAttention(nn.Module):
    """
    Computes a weighted average across the 12 leads dynamically.
    Allows the model to focus on the anatomically relevant leads for each encounter.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, lead_embeddings):
        # lead_embeddings: (B, 12, feature_dim)
        scores = self.attention(lead_embeddings) # (B, 12, 1)
        weights = F.softmax(scores, dim=1) # (B, 12, 1)
        
        # Weighted sum of lead embeddings
        context = torch.sum(weights * lead_embeddings, dim=1) # (B, feature_dim)
        return context, weights.squeeze(-1)

class AnatomicalRegionalAttention(nn.Module):
    """
    Computes attention across clinical anatomical regions rather than independent leads.
    Regions:
      - Inferior: II, III, aVF
      - Anterior/Septal: V1, V2, V3, V4
      - Lateral: I, aVL, V5, V6
      - aVR is isolated
    
    Uses a soft, learnable routing matrix initialized with anatomical priors to allow
    for patient-specific adaptations (e.g. electrode misplacement, atypical presentation).
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Learnable routing matrix: (4 regions, 12 leads)
        self.region_routing = nn.Parameter(torch.zeros(4, 12))
        
        # Initialize with strong anatomical priors
        # Lead Indices: I:0, II:1, III:2, aVR:3, aVL:4, aVF:5, V1:6, V2:7, V3:8, V4:9, V5:10, V6:11
        with torch.no_grad():
            # 0: Inferior
            self.region_routing[0, [1, 2, 5]] = 2.0
            
            # 1: Anterior/Septal
            self.region_routing[1, [6, 7, 8, 9]] = 2.0
            
            # 2: Lateral
            self.region_routing[2, [0, 4, 10, 11]] = 2.0
            
            # 3: aVR
            self.region_routing[3, 3] = 2.0
            
        self.region_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, lead_embeddings):
        # lead_embeddings: (B, 12, feature_dim)
        
        # Step 1: Soft Regional Routing
        # Convert routing logits to probabilities over leads FOR EACH region
        routing_probs = F.softmax(self.region_routing, dim=1) # (4, 12)
        
        # Compute regional embeddings
        # (B, feature_dim, 12) @ (12, 4) -> (B, feature_dim, 4) -> (B, 4, feature_dim)
        regional_embeddings = torch.matmul(
            lead_embeddings.transpose(1, 2), 
            routing_probs.t()
        ).transpose(1, 2)
        
        # Step 2: Attention across Regions
        scores = self.region_attention(regional_embeddings) # (B, 4, 1)
        region_weights = F.softmax(scores, dim=1) # (B, 4, 1)
        
        # Weighted sum of regional embeddings
        context = torch.sum(region_weights * regional_embeddings, dim=1) # (B, feature_dim)
        
        # Compute effective lead weights for interpretability and entropy regularization
        # region_weights: (B, 4, 1)
        # routing_probs: (1, 4, 12)
        effective_lead_weights = torch.sum(
            region_weights * routing_probs.unsqueeze(0), 
            dim=1
        ) # (B, 12)
        
        return context, effective_lead_weights

class LeadAwareHybridExtractor(nn.Module):
    """
    Stage 1: Hybrid Encoder
    Combines independent lead-wise morphological extraction (with spatial attention)
    with the traditional global 12-lead fusion branch.
    """
    def __init__(
        self, 
        in_channels: int = ECG_LEADS, 
        ecg_length: int = ECG_LENGTH, 
        dropout: float = DROPOUT_RATE
    ):
        super().__init__()
        
        # ── Branch A: Lead-Aware Local Morphology (Shared Weights) ──
        # Input: (B, 1, 12, 5000) - treating leads as the "H" dimension
        # The Conv2d with kernel (1, K) slides ONLY over the time dimension,
        # but does it simultaneously and independently for all 12 leads.
        self.local_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Pool across time (W), leaving the 12 leads intact (H)
            nn.AdaptiveMaxPool2d((in_channels, 1)) 
        )
        
        self.spatial_attention = SpatialLeadAttention(feature_dim=64)
        
        # ── Branch B: Global Rhythm/Morphology ──
        # Reuses the exact current extractor to maintain global rhythm context
        self.global_branch = EarlyFusionFeatureExtractor(
            in_channels=in_channels,
            ecg_length=ecg_length,
            dropout=dropout
        ) # Outputs 128D
        
        # ── Fusion ──
        # Local (64) + Global (128) = 192D
        self.output_dim = 64 + 128
        
        # Projection to keep output dim consistent with existing Temporal GRU (128D)
        # This makes it a seamless drop-in replacement
        self.projector = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Save for interpretability (which lead was attended to at this timestep)
        self.last_spatial_weights = None

    def forward(self, x):
        # x: (B, 12, 5000)
        
        # -- Local Branch --
        x_2d = x.unsqueeze(1) # (B, 1, 12, 5000)
        local_feats = self.local_branch(x_2d) # (B, 64, 12, 1)
        
        # Reshape to (B, 12, 64) for attention
        local_feats = local_feats.squeeze(-1).transpose(1, 2) 
        
        # -- Global Branch --
        global_context = self.global_branch(x) # (B, 128)
        
        # Return both so the Temporal Model can compute cross-timestep Lead-Deltas!
        return local_feats, global_context
