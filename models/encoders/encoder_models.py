import torch
import torch.nn as nn
from models.layers.encoder_layers import (
    MambaEncoderLayer, 
    MMMambaEncoderLayer,
    TransformerEncoderLayer,
    MMTransformerEncoderLayer
)


class EncoderSSM(nn.Module):
    """
    State Space Model (SSM) encoder for single modality sequences.
    """
    def __init__(
            self,
            num_layers,
            input_size,
            output_sizes=[256, 512, 512],
            d_ffn=1024,
            activation='Swish',
            dropout=0.0,
            causal=False,
            mamba_config=None
    ):
        """
        Initialize the SSM encoder.
        
        Args:
            num_layers: Number of encoder layers
            input_size: Input dimension
            output_sizes: List of output dimensions for each layer
            d_ffn: Feed-forward network dimension
            activation: Activation function
            dropout: Dropout probability
            causal: Whether to use causal attention
            mamba_config: Configuration for Mamba SSM
        """
        super().__init__()
        
        # Ensure output_sizes has the right length
        assert len(output_sizes) > 0, "Output sizes must be provided"
        
        if len(output_sizes) < num_layers:
            # If fewer output sizes than layers, repeat the last one
            output_sizes = output_sizes + [output_sizes[-1]] * (num_layers - len(output_sizes))
        elif len(output_sizes) > num_layers:
            # If more output sizes than layers, truncate
            output_sizes = output_sizes[:num_layers]
            
        # Input projection if needed
        if input_size != output_sizes[0]:
            self.input_proj = nn.Linear(input_size, output_sizes[0])
        else:
            self.input_proj = nn.Identity()
            
        # Build encoder layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                MambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    activation=activation,
                    dropout=dropout,
                    causal=causal,
                    mamba_config=mamba_config.copy()  # Use copy to avoid modifying the original
                )
            )

    def forward(
            self,
            x,
            inference_params=None,
    ):
        """
        Forward pass through the SSM encoder.
        
        Args:
            x: Input tensor
            inference_params: Optional inference parameters
            
        Returns:
            Encoded output
        """
        # Apply input projection
        x = self.input_proj(x)
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, inference_params)
            
        return x


class CrossModalSSM(nn.Module):
    """
    Cross-modal State Space Model (SSM) encoder for two modalities.
    """
    def __init__(
            self,
            num_layers,
            input_size,
            output_sizes=[256, 512, 512],
            d_ffn=1024,
            activation='Swish',
            dropout=0.0,
            kernel_size=3,  # Unused but kept for compatibility
            causal=False,
            mamba_config=None
    ):
        """
        Initialize the cross-modal SSM encoder.
        
        Args:
            num_layers: Number of encoder layers
            input_size: Input dimension
            output_sizes: List of output dimensions for each layer
            d_ffn: Feed-forward network dimension
            activation: Activation function
            dropout: Dropout probability
            kernel_size: Unused (kept for backwards compatibility)
            causal: Whether to use causal attention
            mamba_config: Configuration for Mamba SSM
        """
        super().__init__()
        
        # Ensure output_sizes has the right length
        assert len(output_sizes) > 0, "Output sizes must be provided"
        
        if len(output_sizes) < num_layers:
            # If fewer output sizes than layers, repeat the last one
            output_sizes = output_sizes + [output_sizes[-1]] * (num_layers - len(output_sizes))
        elif len(output_sizes) > num_layers:
            # If more output sizes than layers, truncate
            output_sizes = output_sizes[:num_layers]
            
        # Input projections if needed
        if isinstance(input_size, (list, tuple)):
            a_input_size, v_input_size = input_size
            if a_input_size != output_sizes[0]:
                self.a_input_proj = nn.Linear(a_input_size, output_sizes[0])
            else:
                self.a_input_proj = nn.Identity()
                
            if v_input_size != output_sizes[0]:
                self.v_input_proj = nn.Linear(v_input_size, output_sizes[0])
            else:
                self.v_input_proj = nn.Identity()
        else:
            # Same input size for both modalities
            if input_size != output_sizes[0]:
                self.a_input_proj = nn.Linear(input_size, output_sizes[0])
                self.v_input_proj = nn.Linear(input_size, output_sizes[0])
            else:
                self.a_input_proj = nn.Identity()
                self.v_input_proj = nn.Identity()
                
        # Build encoder layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                MMMambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    activation=activation,
                    dropout=dropout,
                    causal=causal,
                    mamba_config=mamba_config.copy()  # Use copy to avoid modifying the original
                )
            )

    def forward(
            self,
            a_x, v_x,
            a_inference_params=None,
            v_inference_params=None
    ):
        """
        Forward pass through the cross-modal SSM encoder.
        
        Args:
            a_x: Audio modality input tensor
            v_x: Visual modality input tensor
            a_inference_params: Optional audio inference parameters
            v_inference_params: Optional visual inference parameters
            
        Returns:
            Tuple of encoded outputs for both modalities
        """
        # Apply input projections
        a_x = self.a_input_proj(a_x)
        v_x = self.v_input_proj(v_x)
        
        # Process through encoder layers
        for layer in self.layers:
            a_x, v_x = layer(a_x, v_x, a_inference_params, v_inference_params)
            
        return a_x, v_x 