from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F

from network import Decoder, Encoder

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
       # convert inputs from NCHW -> NHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # .contiguous() after permute ensures proper memory layout for view operations
        input_shape = inputs.shape  # [N, H, W, C]

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  #[N, H, W, C] -> [N*H*W, C]


        # self._embedding.weight.shape = [E, C]
        # - E: num_embeddings (size of codebook)
        # - C: embedding_dim (dimensionality of each code vector)

        # Calculate squared Euclidean distances between all input vectors and codebook entries
        # (a-b)^2 = a^2 + b^2 -2ab
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)  # sum( [N*H*W, C] ** 2) -> [N*H*W, 1]  
                    + torch.sum(self._embedding.weight**2, dim=1)  # sum( [E, C] ** 2 )    -> [E]
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t())) # [N*H*W, C] @ [C, E] -> [N*H*W, E]

        # distances.shape = [N*H*W, E]
        # So for each N*W*H, there are E distances to every vector in the coodbook

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) #  torch.argmin([N*H*W, E], 1) -> [N*H*W] -> [N*H*W, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device) # [N*H*W, E]
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encoding

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # [N*H*W, E] @ [E, C] -> [N*H*W, C] ->  [N, H, W, C]

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from NHWC -> NCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from NCHW -> NHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # .contiguous() after permute ensures proper memory layout for view operations
        input_shape = inputs.shape  # [N, H, W, C]

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  #[N, H, W, C] -> [N*H*W, C]


        # self._embedding.weight.shape = [E, C]
        # - E: num_embeddings (size of codebook)
        # - C: embedding_dim (dimensionality of each code vector)

        # Calculate squared Euclidean distances between all input vectors and codebook entries
        # (a-b)^2 = a^2 + b^2 -2ab
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)  # sum( [N*H*W, C] ** 2) -> [N*H*W, 1]  
                    + torch.sum(self._embedding.weight**2, dim=1)  # sum( [E, C] ** 2 )    -> [E]
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t())) # [N*H*W, C] @ [C, E] -> [N*H*W, E]
        
        # distances.shape = [N*H*W, E], for each N*W*H, there are E distances to every vector in the coodbook

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) #  torch.argmin([N*H*W, E], 1) -> [N*H*W] -> [N*H*W, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device) # [N*H*W, E]
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encoding

        """
        # flat_input  [N*H*W, C]                   Codebook  [E, C]
        [[1, 2],                                   [[8, 9],
         [5, 6]]                                    [2, 3],
                                                    [6, 7]]

        # Squared flat_input  [N*H*W, 1]           Squared Codebook [E]
        [[1² + 2²], = [[5],                        [8² + 9²,  =  [145, 13, 85]
         [5² + 6²]]    [61]]                        2² + 3²,
                                                    6² + 7²]

        # Matrix multiplication   [N*H*W, C] @ [C, E] -> [N*H*W, E]
        [[1, 2],   @   [[8, 2, 6], = [[26,  8, 20],
         [5, 6]]        [9, 3, 7]]    [94, 28, 72]]

        # Scaled cross terms
        2 x [[26,  8, 20],   = [[52,  16, 40 ], 
             [94, 28, 72]]      [188, 56, 144]]

        # Squared distances                                [N*H*W, E]
        [[ 5, 5, 5], + [[145,13,85], - [[52,  16, 40 ],  = [[98,  2, 50]
         [61,61,61]]    [145,13,85]]    [188, 56, 144]]     [18, 18,  2]]

        # Encoding indices [N*H*W, 1]          # One-hot encodings [N*H*W, E]
        [[1],                                  [[0,1,0],
         [2]]                                   [0,0,1]]

        # Quantized output
        [[0, 1, 0],  @   [[8, 9],   =   [[2, 3],
         [0, 0, 1]]       [2, 3],        [6 ,7]]
                          [6, 7]]
        """

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # [N*H*W, E] @ [E, C] -> [N*H*W, C] ->  [N, H, W, C]

        # Use EMA to update the embedding vectors
        if self.training:
            """
            # _ema_cluster_size [E] records the number of times each embedding is selected 
            # One-hot encodings [N*H*W, E]           # torch.sum(encodings, 0) [E]
               [[0,1,0],                               [0,1,1]
                [0,0,1]]
            """
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            """
            # Laplace smoothing (additive smoothing) adjusts probability estimates by adding a small value (ε)
            # to each possible outcome to avoid zero probabilities for events that haven’t been observed.
            P'(x) = (count(x) + ε) / (total_count + K * ε)
            Here we multiply k in the end to ensure sum of _ema_cluster_size equals to the total times of selection
            """
            k = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (k + self._num_embeddings * self._epsilon) * k)

            """
            # transposed encodings [E, N*H*W]           # flat_input [N*H*W, C]     dw [E, C]
               [[0,0],                          @        [[1, 2],                =    [[0, 0],
                [1,0],                                    [5, 6]]                      [1, 2],
                [0,1]]                                                                 [5, 6]]
            # Each row of dw is the sum of inputs mapped to an embedding.
            """
            dw = torch.matmul(encodings.t(), flat_input)

            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        # .detach() Stops the gradient from propagating back through the quantization process into the codebook embeddings (self._embedding.weight).
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # Difference between encoder output and quantized
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from NHWC -> NCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    


class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity