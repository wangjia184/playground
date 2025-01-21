from diffusers import UNet2DModel

def create_improved_unet():
    """
    Create an improved UNet2DModel with attention mechanisms and increased complexity.
    """
    model = UNet2DModel(
        sample_size=28,  # Input sample size (height and width)
        in_channels=1,   # Number of input channels
        out_channels=1,  # Number of output channels
        block_out_channels=(32, 64, 128),  # Channels for each block
        layers_per_block=3,  # Number of ResNet layers per block
        norm_num_groups=8,   # Group normalization for stability
        down_block_types=(
            "DownBlock2D",  # First block without attention
            "AttnDownBlock2D",  # Second block with attention
            "AttnDownBlock2D",  # Third block with attention
        ),
        up_block_types=(
            "AttnUpBlock2D",  # First up block with attention
            "AttnUpBlock2D",  # Second up block with attention
            "UpBlock2D",  # Last block without attention
        ),
        act_fn="silu",  # Activation function (SiLU is default)
        attention_head_dim=16,  # Attention head dimension
        time_embedding_type="positional",  # Time embedding type
        dropout=0.1,  # Dropout for regularization
    )
    return model