import math
import torch
from torch import nn


class VAETransformerEncoder(nn.Module):
    """
        Transformer encoder class
    """
    def __init__(self, num_layers, dim_model, num_head, dim_feedforward, dim_vae_latent, dropout=0.1,
                 activation='relu'):
        super(VAETransformerEncoder, self).__init__()
        self.num_layers = num_layers              # Number of sub-encoder-layers in the encoder
        self.dim_model = dim_model                # Number of expected features in the input
        self.num_head = num_head                  # Number of heads for the multi-head-attention model
        self.dim_feedforward = dim_feedforward    # The dimension of the feedforward network model
        self.activation = activation              # Activation function of the intermediate layer
        self.dim_vae_latent = dim_vae_latent      # Dimension of latent vector

        # TransformerEncoderLayer is made up of self-attn and feedforward network
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(dim_model, num_head, dim_feedforward, dropout,
                                                                    activation)

        # TransformerEncoder is a stack of N encoder layers
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, self.num_layers,
                                                         enable_nested_tensor=False)

        # Variational Auto Encoder latent space
        self.vae_mu = nn.Linear(dim_model, dim_vae_latent)
        self.vae_logvar = nn.Linear(dim_model, dim_vae_latent)

    def forward(self, x, padding_mask=None):
        # Pass input through the transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Extract hidden state from the first position, assuming a CLS token first
        hidden_out = output[0, :, :]

        # Different perspective
        # hidden_out = torch.mean(output, dim=0)  # Shape: [batch_size, d_model]

        # Compute the mean (mu) and log-variance (logvar) for the VAE latent space
        mu = self.vae_mu(hidden_out)
        logvar = self.vae_logvar(hidden_out)

        return hidden_out, mu, logvar


class VAETransformerDecoder(nn.Module):
    def __init__(self, num_layers, dim_model, num_head, dim_feedforward, dim_condition_embedding, dropout=0.1,
                 activation='relu'):
        super(VAETransformerDecoder, self).__init__()
        self.num_layers = num_layers                               # Number of sub-encoder-layers in the encoder
        self.dim_model = dim_model                                 # Number of expected features in the input
        self.num_head = num_head                                   # Number of heads for the multi-head-attention model
        self.dim_feedforward = dim_feedforward                     # The dimension of the feedforward network model
        self.dim_condition_embedding = dim_condition_embedding     # Dimensions of the conditional embedding
        self.dropout = dropout                                     # Dropout value
        self.activation = activation                               # Activation function of the intermediate layer

        # Using in-attention conditioning
        self.condition_embedding_proj = nn.Linear(dim_condition_embedding, dim_model, bias=False)

        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(nn.TransformerEncoderLayer(dim_model, num_head, dim_feedforward, dropout,
                                                                  activation))

    def forward(self, x, condition_embedding, verbose=False):
        attn_mask = generate_causal_mask(x.size(0)).to(x.device)
        if verbose:
            print("Attention mask size: ", attn_mask.size())

        # Using in-attention conditioning
        condition_embedding = self.condition_embedding_proj(condition_embedding)

        out = x
        for i in range(self.num_layers):
            # Add in-attention conditioning to the input, and subsequent outputs
            out += condition_embedding

            # Get decoder layer output
            out = self.decoder_layers[i](out, src_mask=attn_mask)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_pos=16384):
        super(PositionalEncoding, self).__init__()
        self.dim_model = dim_model          # Model dimensions
        self.dropout = nn.Dropout(dropout)  # Dropout module
        self.max_pos = max_pos              # Maximum position

        # Calculate positional encodings
        pe = torch.zeros(max_pos, 1, dim_model)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, batch=True):
        # Slicing the positional encodings up to the input sequence length
        pos_encoding = self.pe[:x.size(0), :, :]

        # Match dimensions to avoid funny broadcasting
        if batch:
            pos_encoding = pos_encoding.unsqueeze(2)

        # Adding positional encodings to the input tensor
        x = self.dropout(x) + pos_encoding

        # Apply dropout
        return x


class EmbeddingWithProjection(nn.Module):
    def __init__(self, num_embeddings, dim_embedding, projected_dimensions):
        super(EmbeddingWithProjection, self).__init__()

        self.num_embeddings = num_embeddings                # Size of the dictionary of embeddings, in this case tokens
        self.dim_embedding = dim_embedding                  # Size of each embedding vector
        self.projected_dimensions = projected_dimensions    # Size of the dimensions of the output
        self.embedding_scale = projected_dimensions ** 0.5  # Square root scaling factor

        # Create the embedding layer
        self.embedding_lookup = nn.Embedding(num_embeddings, dim_embedding)

        # If the number of projected dimensions is not equal to the number of embedding dimensions use a linear layer
        # to map the embedding dimensions with the projected dimensions.
        if projected_dimensions != dim_embedding:
            self.embedding_projection = nn.Linear(dim_embedding, projected_dimensions, bias=False)
        else:
            self.embedding_projection = None

    def forward(self, x):
        # Run the input through the embedding layer
        input_embedding = self.embedding_lookup(x)

        # If necessary, adjust the dimensions of the output by projecting the embeddings through a linear layer
        if self.embedding_projection is not None:
            input_embedding = self.embedding_projection(input_embedding)

        # Return a scaled embedding
        return input_embedding.mul_(self.embedding_scale)


def init_weights(module, verbose=False):
    classname = module.__class__.__name__

    # Print the name of the module being initialised
    if verbose:
        print(f'[{classname}] initialising ...')

    # Linear layers
    if classname.find('Linear') != -1:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, 0.0, 0.01)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    # Embedding layers
    elif classname.find('Embedding') != -1:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, 0.0, 0.01)
    # Layer normalisation
    elif classname.find('LayerNorm') != -1:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, 0.0, 0.01)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    # Initialise GRU layers
    elif classname.find('GRU') != -1:
        for param in module.parameters():
            # For weights
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param, 0.01)
            # For bias
            else:
                nn.init.constant_(param, 0.0)
    # If module not recognised print it
    elif verbose:
        print(f'[{classname}] not initialised!')


def generate_causal_mask(seq_len):
    # Create an upper triangular matrix of ones
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)

    # Convert the boolean mask to float and apply masking
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    # Ensure the mask is not part of the computation graph
    mask.requires_grad = False

    return mask
