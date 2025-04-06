import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.encoders.encoder_models import EncoderSSM, CrossModalSSM
from utils.activations import diff_softmax


class DynMultiDep(BaseModel):
    """
    Dynamic Multi-modal Dependency model for multimodal deception detection.
    """
    def __init__(
            self,
            audio_input_size=161,
            video_input_size=161,
            mm_input_size=128,
            mm_output_sizes=[256, 64],
            d_ffn=1024,
            num_layers=8,
            dropout=0.1,
            activation='Swish',
            causal=False,
            mamba_config=None,
            temp=1.0,
            hard_gate=True
    ):
        """
        Initialize the DynMultiDep model.
        
        Args:
            audio_input_size: Dimension of audio input features
            video_input_size: Dimension of video input features
            mm_input_size: Dimension of multimodal features
            mm_output_sizes: Output dimensions for multimodal encoder
            d_ffn: Feed-forward network dimension
            num_layers: Number of encoder layers
            dropout: Dropout probability
            activation: Activation function
            causal: Whether to use causal attention
            mamba_config: Configuration for Mamba SSM
            temp: Temperature for differentiable softmax
            hard_gate: Whether to use hard gating
        """
        super().__init__()
        
        # Default Mamba config if none provided
        if mamba_config is None:
            mamba_config = {
                'd_state': 16,
                'expand': 2,
                'd_conv': 4,
                'bidirectional': True
            }
        
        # Model parameters
        self.temp = temp
        self.hard_gate = hard_gate
        self.mm_output_dim = mm_output_sizes[-1]
        
        # Create cross-modal encoder for multimodal interaction
        self.mm_encoder = CrossModalSSM(
            num_layers=num_layers,
            input_size=[audio_input_size, video_input_size],
            output_sizes=mm_output_sizes,
            d_ffn=d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config
        )
        
        # Create gate mechanism for dynamic modality weighting
        self.modality_gate = nn.Sequential(
            nn.Linear(mm_output_sizes[-1] * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Final classifier
        self.classifier_layer = nn.Sequential(
            nn.Linear(mm_output_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.reset_weights()
        
    def reset_weights(self):
        """
        Initialize model weights using Xavier/Glorot initialization.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'mamba' not in name:  # Don't reset Mamba module weights
                nn.init.xavier_uniform_(p)
                
    def feature_extractor(
            self,
            x,
            padding_mask=None,
            a_inference_params=None,
            v_inference_params=None
    ):
        """
        Extract features from multimodal input using the cross-modal encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim * 2]
                where the feature dimension contains concatenated audio and video features
            padding_mask: Optional mask for padding
            a_inference_params: Optional audio inference parameters
            v_inference_params: Optional visual inference parameters
            
        Returns:
            Feature representation with dynamic modality weighting
        """
        batch_size = x.shape[0]
        
        # Split input into audio and video modalities
        # We assume the first half of the feature dim is audio, second half is video
        feature_dim = x.shape[2] // 2
        a_x = x[:, :, :feature_dim]  # Audio features
        v_x = x[:, :, feature_dim:]  # Video features
        
        # Process through multimodal encoder
        a_features, v_features = self.mm_encoder(
            a_x, v_x, a_inference_params, v_inference_params
        )
        
        # Temporal average pooling
        if padding_mask is not None:
            # Apply mask for proper averaging
            mask_expanded = padding_mask.unsqueeze(-1)
            a_pooled = (a_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            v_pooled = (v_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling if no mask
            a_pooled = a_features.mean(dim=1)
            v_pooled = v_features.mean(dim=1)
        
        # Compute dynamic modality weights
        combined_features = torch.cat([a_pooled, v_pooled], dim=1)
        modality_logits = self.modality_gate(combined_features)
        
        # Apply differentiable softmax for modality weighting
        modality_weights = diff_softmax(
            modality_logits, tau=self.temp, hard=self.hard_gate
        )
        
        # Weight modalities
        weighted_features = (
            a_pooled * modality_weights[:, 0].unsqueeze(1) +
            v_pooled * modality_weights[:, 1].unsqueeze(1)
        )
        
        return weighted_features
        
    def classifier(self, x):
        """
        Classify features.
        
        Args:
            x: Feature tensor
            
        Returns:
            Classification output
        """
        return self.classifier_layer(x)
        
    def forward(self, x, padding_mask=None, a_inference_params=None, v_inference_params=None):
        """
        Forward pass through the DynMultiDep model.
        
        Args:
            x: Input tensor
            padding_mask: Optional mask for padding
            a_inference_params: Optional audio inference parameters
            v_inference_params: Optional visual inference parameters
            
        Returns:
            Model output and any additional losses
        """
        features = self.feature_extractor(
            x, padding_mask, a_inference_params, v_inference_params
        )
        output = self.classifier(features)
        
        # No additional loss in this implementation
        return output, 0.0 