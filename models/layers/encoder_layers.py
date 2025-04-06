import torch
import torch.nn as nn
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.nnet.normalization import LayerNorm
from mamba_ssm import Mamba
from models.mamba_ssm.bimamba import Mamba as BiMamba
from models.mamba_ssm.mm_bimamba import Mamba as MMBiMamba


class MambaEncoderLayer(nn.Module):
    """
    Single modality Mamba encoder layer.
    """
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='Swish',
            dropout=0.0,
            causal=False,
            mamba_config=None
    ):
        """
        Initialize Mamba encoder layer.
        
        Args:
            d_model: Model dimension
            d_ffn: Feed-forward network dimension
            activation: Activation function
            dropout: Dropout probability
            causal: Whether to use causal attention
            mamba_config: Configuration for Mamba SSM
        """
        super().__init__()
        assert mamba_config is not None, "Mamba configuration must be provided"

        # Select activation function
        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        # Save bidirectional config and restore it later
        bidirectional = mamba_config.pop('bidirectional')
        
        # Choose Mamba variant based on directionality
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        
        # Restore bidirectional config
        mamba_config['bidirectional'] = bidirectional

        # Layer normalization and dropout
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        """
        Forward pass through the Mamba encoder layer.
        
        Args:
            x: Input tensor
            inference_params: Optional inference parameters
            
        Returns:
            Encoded output
        """
        # Residual connection with layer norm
        out = x + self.norm1(self.mamba(x, inference_params))
        return out


class MMMambaEncoderLayer(nn.Module):
    """
    Multi-modal Mamba encoder layer for processing two modalities.
    """
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='Swish',
            dropout=0.0,
            causal=False,
            mamba_config=None
    ):
        """
        Initialize multi-modal Mamba encoder layer.
        
        Args:
            d_model: Model dimension
            d_ffn: Feed-forward network dimension
            activation: Activation function
            dropout: Dropout probability
            causal: Whether to use causal attention
            mamba_config: Configuration for Mamba SSM
        """
        super().__init__()
        assert mamba_config is not None, "Mamba configuration must be provided"

        # Select activation function
        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        # Save bidirectional config and restore it later
        bidirectional = mamba_config.pop('bidirectional')
        
        # Choose Mamba variant based on directionality
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        
        # Restore bidirectional config
        mamba_config['bidirectional'] = bidirectional

        # Layer normalization for both modalities
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        # Optional downsampling for audio modality
        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
            self,
            a_x, v_x,
            a_inference_params=None,
            v_inference_params=None
    ):
        """
        Forward pass through the multi-modal Mamba encoder layer.
        
        Args:
            a_x: Audio modality input tensor
            v_x: Visual modality input tensor
            a_inference_params: Optional audio inference parameters
            v_inference_params: Optional visual inference parameters
            
        Returns:
            Tuple of encoded outputs for both modalities
        """
        # Process both modalities through Mamba
        a_out1, v_out1 = self.mamba(a_x, v_x, a_inference_params, v_inference_params)
        
        # Residual connections with layer norm
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)

        return a_out, v_out


class TransformerEncoderLayer(nn.Module):
    """
    Single modality Transformer encoder layer.
    """
    def __init__(
            self,
            input_size,
            output_size,
            d_ffn=2048,
            nhead=8,
            dropout=0.0,
            causal=False,
    ):
        """
        Initialize Transformer encoder layer.
        
        Args:
            input_size: Input dimension
            output_size: Output dimension
            d_ffn: Feed-forward network dimension
            nhead: Number of attention heads
            dropout: Dropout probability
            causal: Whether to use causal attention
        """
        super().__init__()

        # Multi-head attention
        self.mha = MultiheadAttention(
            nhead=nhead,
            d_model=input_size,
            dropout=dropout,
        )
        
        # Feed-forward network
        self.ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=input_size,
            dropout=dropout,
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(input_size, eps=1e-6)
        self.norm2 = LayerNorm(input_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # Optional projection layer if input and output dimensions differ
        if input_size != output_size:
            self.proj = nn.Linear(input_size, output_size)
        else:
            self.proj = None

    def forward(self, x):
        """
        Forward pass through the Transformer encoder layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded output
        """
        # Self-attention with residual connection and layer norm
        out = self.mha(x, x, x)[0]
        out = self.norm1(x + self.dropout(out))
        
        # Feed-forward network with residual connection and layer norm
        ffn_out = self.ffn(out)
        out = self.norm2(out + self.dropout(ffn_out))

        # Optional projection
        if self.proj is not None:
            out = self.proj(out)
        return out


class MMTransformerEncoderLayer(nn.Module):
    """
    Multi-modal Transformer encoder layer for processing two modalities.
    """
    def __init__(
            self,
            input_size,
            output_size,
            d_ffn=2048,
            nhead=8,
            dropout=0.0,
            causal=False,
    ):
        """
        Initialize multi-modal Transformer encoder layer.
        
        Args:
            input_size: Input dimension
            output_size: Output dimension
            d_ffn: Feed-forward network dimension
            nhead: Number of attention heads
            dropout: Dropout probability
            causal: Whether to use causal attention
        """
        super().__init__()

        # Audio modality components
        self.a_mha = MultiheadAttention(
            nhead=nhead,
            d_model=input_size,
            dropout=dropout,
        )
        self.a_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=input_size,
            dropout=dropout,
        )
        self.a_norm1 = LayerNorm(input_size, eps=1e-6)
        self.a_norm2 = LayerNorm(input_size, eps=1e-6)

        # Visual modality components
        self.v_mha = MultiheadAttention(
            nhead=nhead,
            d_model=input_size,
            dropout=dropout,
        )
        self.v_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=input_size,
            dropout=dropout,
        )
        self.v_norm1 = LayerNorm(input_size, eps=1e-6)
        self.v_norm2 = LayerNorm(input_size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        # Optional projection layers if input and output dimensions differ
        if input_size != output_size:
            self.a_proj = nn.Linear(input_size, output_size)
            self.v_proj = nn.Linear(input_size, output_size)
        else:
            self.a_proj = None
            self.v_proj = None

    def forward(self, xa, xv):
        """
        Forward pass through the multi-modal Transformer encoder layer.
        
        Args:
            xa: Audio modality input tensor
            xv: Visual modality input tensor
            
        Returns:
            Tuple of encoded outputs for both modalities
        """
        # Audio path
        a_out = self.a_mha(xa, xa, xa)[0]
        a_out = self.a_norm1(xa + self.dropout(a_out))
        a_ffn_out = self.a_ffn(a_out)
        a_out = self.a_norm2(a_out + self.dropout(a_ffn_out))

        # Visual path
        v_out = self.v_mha(xv, xv, xv)[0]
        v_out = self.v_norm1(xv + self.dropout(v_out))
        v_ffn_out = self.v_ffn(v_out)
        v_out = self.v_norm2(v_out + self.dropout(v_ffn_out))

        # Optional projections
        if self.a_proj is not None:
            a_out = self.a_proj(a_out)
        if self.v_proj is not None:
            v_out = self.v_proj(v_out)

        return a_out, v_out 