from diffusers import UNet2DModel



def create_unet():
    # Initialize the UNet2DModel with parameters similar to your current Unet
    model = UNet2DModel(
        sample_size=(64, 64),  # Replace with the appropriate sample size
        in_channels=3,        # Number of input channels (e.g., RGB images)
        out_channels=3,       # Number of output channels
        block_out_channels=(64, 128, 256, 512),  # Adjust based on your model's depth
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        layers_per_block=2,    # Number of layers per block
        act_fn="silu",         # Activation function
        norm_num_groups=32,    # Number of groups for normalization
        attention_head_dim=8,  # Attention head dimension
        dropout=0.0            # Dropout probability
    )

    return model