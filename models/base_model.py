import abc
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for all models in the project.
    Defines the basic interface that all models should implement.
    """
    
    def __init__(self):
        """
        Initialize the base model.
        """
        super().__init__()

    @abc.abstractmethod
    def feature_extractor(self, x, padding_mask=None):
        """
        Extract features from input data.
        
        Args:
            x: Input tensor
            padding_mask: Optional mask for padding
            
        Returns:
            Feature representation
        """
        pass

    @abc.abstractmethod
    def classifier(self, x):
        """
        Classify features into output classes/values.
        
        Args:
            x: Feature tensor
            
        Returns:
            Classification output
        """
        pass

    def forward(self, x, padding_mask=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            padding_mask: Optional mask for padding
            
        Returns:
            Model output and any additional losses
        """
        x = self.feature_extractor(x, padding_mask)
        output = self.classifier(x)
        # Default implementation returns no additional loss
        return output, 0.0


class TemporalMeanNet(BaseModel):
    """
    Simple model that applies temporal mean pooling followed by a MLP classifier.
    """
    
    def __init__(self, input_dim=161, hidden_sizes=[256, 128], dropout=0.5):
        """
        Initialize the temporal mean network.
        
        Args:
            input_dim: Dimension of input features
            hidden_sizes: List of hidden layer sizes for MLP
            dropout: Dropout probability
        """
        super().__init__()
        self.mlp = nn.Sequential()
        last_dim = input_dim
        
        # Build MLP layers
        for h in hidden_sizes:
            self.mlp.append(nn.Linear(last_dim, h))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
            last_dim = h
            
        self.output_layer = nn.Linear(last_dim, 1)

    def feature_extractor(self, x, padding_mask=None):
        """
        Extract features by applying temporal mean pooling.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            padding_mask: Optional mask for padding
            
        Returns:
            Mean-pooled features
        """
        # Simple temporal mean pooling
        return x.mean(dim=1)

    def classifier(self, x):
        """
        Classify features.
        
        Args:
            x: Feature tensor
            
        Returns:
            Classification output
        """
        return self.output_layer(self.mlp(x)) 