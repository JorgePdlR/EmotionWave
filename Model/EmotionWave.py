import torch
from torch import nn
import torch.nn.functional as F
from .VAETransformer import (VAETransformerEncoder, VAETransformerDecoder, init_weights, PositionalEncoding,
                            EmbeddingWithProjection, generate_causal_mask)


class EmotionWave(nn.Module):
    def __init__(self,
                 encoder_num_layers, encoder_dim_model, encoder_num_head, encoder_dim_feedforward,  # Encoder
                 dim_vae_latent,  # VAE latent space
                 dim_embedding, num_embeddings,  # Embedding module
                 decoder_num_layers, decoder_dim_model, decoder_num_head, decoder_dim_feedforward,  # Decoder
                 valence_num_cls=8, valence_dim_embeddings=32,  # Valence conditioning
                 encoder_dropout=0.1, encoder_activation='relu', decoder_dropout=0.1, decoder_activation='relu',
                 valence_cls=True):
        super(EmotionWave, self).__init__()
        # Encoder
        self.encoder_num_layers = encoder_num_layers
        self.encoder_dim_model = encoder_dim_model
        self.encoder_num_head = encoder_num_head
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.encoder_dropout = encoder_dropout
        self.encoder_activation = encoder_activation

        # Decoder
        self.decoder_num_layers = decoder_num_layers
        self.decoder_dim_model = decoder_dim_model
        self.decoder_num_head = decoder_num_head
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.decoder_dropout = decoder_dropout
        self.decoder_activation = decoder_activation

        # VAE
        self.dim_vae_latent = dim_vae_latent

        # Valence conditioning
        self.valence_dim_embeddings = valence_dim_embeddings
        self.valence_num_cls = valence_num_cls
        self.valence_cls = valence_cls

        # Embeddings
        self.dim_embedding = dim_embedding
        self.num_embedding = num_embeddings

        # Modules
        # Embedding
        self.input_embedding = EmbeddingWithProjection(self.num_embedding, self.dim_embedding, self.encoder_dim_model)
        # Positional encoder
        self.positional_encoder = PositionalEncoding(self.dim_embedding, max_pos=10000)
        # Encoder
        self.encoder = VAETransformerEncoder(self.encoder_num_layers, self.encoder_dim_model, self.encoder_num_head,
                                             self.encoder_dim_feedforward, self.dim_vae_latent, self.encoder_dropout,
                                             self.encoder_activation)

        # Decoder
        # Use valence cls conditioning if provided
        if self.valence_cls:
            self.decoder = VAETransformerDecoder(self.decoder_num_layers, self.decoder_dim_model, self.decoder_num_head,
                                                 self.decoder_dim_feedforward,
                                                 self.dim_vae_latent + self.valence_dim_embeddings,
                                                 self.decoder_dropout, self.decoder_activation)

            self.valence_embedding = EmbeddingWithProjection(self.valence_num_cls, self.valence_dim_embeddings,
                                                             self.valence_dim_embeddings)

        else:
            self.decoder = VAETransformerDecoder(self.decoder_num_layers, self.decoder_dim_model, self.decoder_num_head,
                                                 self.decoder_dim_feedforward, self.dim_vae_latent,
                                                 self.decoder_dropout, self.decoder_activation)

            self.valence_embedding = None

        # Linear layer to match the decoder dimensions to the embeddings vocabulary
        self.decoder_out_projection = nn.Linear(self.decoder_dim_model, self.num_embedding)

        # Initialise weights
        self.apply(init_weights)

    def reparameterise(self, mu, logvar, use_sampling=True, sampling_var=1.0):
        # Re-parametrisation is done to make sampling differentiable. Instead of
        # computing the derivative of the normal distribution, the random variable
        # z can be differentiated by expressing it as (epsilon * std) + mu.
        # Calculate the standard deviation
        std = torch.exp(0.5 * logvar).to(mu.device)

        # Sample from a normal distribution
        if use_sampling:
            eps = torch.randn_like(std).to(mu.device) * sampling_var
        # Zero vector for deterministic output
        else:
            eps = torch.zeros_like(std).to(mu.device)

        # Return re-parametrised latent variable
        return eps * std + mu

    def get_sampled_latent(self, x, padding_mask=None, use_sampling=False, sampling_var=0.):
        # Input embedding
        input_embedding = self.input_embedding(x)

        # Embedding positional encoding
        input_encoding = self.positional_encoder(input_embedding, batch=False)

        # Encode to get mu and log-variance
        _, mu, logvar = self.encoder(input_encoding, padding_mask=padding_mask)

        # Reshape for re-parametrisation
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        # Re-parametrise to get latent variable
        vae_latent = self.reparameterise(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

        return vae_latent

    def generate(self, x, condition_embedding, valence_cls=None, keep_last_only=True):

        print("input", x.shape)

        # Input embedding
        input_embedding = self.input_embedding(x)

        print("After embedding", input_embedding.shape)

        # Embedding positional encoding
        input_decoder = self.positional_encoder(input_embedding, batch=False)

        print("After encoder", input_decoder.shape)

        # Concatenate to the conditional embedding valence conditioning, if provided
        if valence_cls is not None:
            decoder_valence_embedding = self.valence_embedding(valence_cls)
            decoder_condition_embedding = torch.cat([condition_embedding, decoder_valence_embedding], dim=-1)
        else:
            decoder_condition_embedding = condition_embedding

        print("Extra, condition embedding", decoder_condition_embedding.shape)

        # Decode the input
        out = self.decoder(input_decoder, decoder_condition_embedding)
        # Project decoder output
        out = self.decoder_out_projection(out)

        # Keep only last output
        if keep_last_only:
            out = out[-1, ...]

        return out

    def forward(self, encoder_x, decoder_x, decoder_x_bar_position, valence_cls=None,padding_mask=None, verbose=False):
        # Get shape information
        encoder_batch_size, encoder_num_bars = encoder_x.size(1), encoder_x.size(2)
        if verbose:
            print("Shape of input encoder:", encoder_x.shape)
            print("Encoder batch size", encoder_batch_size, ", number of bars", encoder_num_bars)
            print("Shape of input decoder", decoder_x.shape)

        ### Encoder input processing ###
        # Encoder input embedding
        encoder_input_embedding = self.input_embedding(encoder_x)
        if verbose:
            print("Encoder input embedding", encoder_input_embedding.shape)

        encoder_input_embedding = encoder_input_embedding.reshape(encoder_x.size(0), -1,
                                                                  encoder_input_embedding.size(-1))
        if verbose:
            print("Encoder reshaped input embedding", encoder_input_embedding.shape)

        # Encoder positional encoding
        encoder_input_encoding = self.positional_encoder(encoder_input_embedding, batch=False)

        if verbose:
            print("Encoder input encoding", encoder_input_encoding.shape)

        ### Decoder input processing ###
        # Decoder input embedding
        decoder_input_embedding = self.input_embedding(decoder_x)

        if verbose:
            print("Decoder input embedding", decoder_input_embedding.shape)

        # Decoder positional encoding
        decoder_input_encoding = self.positional_encoder(decoder_input_embedding, batch=False)

        ### Encoder processing ###
        if verbose and padding_mask is not None:
            print("Padding mask dimensions", padding_mask.shape)

        # Add padding for those indices bigger than the sequence length per bar
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        if verbose and padding_mask is not None:
            print("Padding mask reshaped dimensions", padding_mask.shape)

        _, mu, logvar = self.encoder(encoder_input_encoding, padding_mask=padding_mask)
        vae_latent_space = self.reparameterise(mu, logvar)

        if verbose:
            print("VAE latent space", vae_latent_space.shape)

        vae_latent_space_reshaped = vae_latent_space.reshape(encoder_batch_size, encoder_num_bars, -1)

        if verbose:
            print("VAE latent space reshaped", vae_latent_space_reshaped.shape)

        ### Decoder processing ###

        decoder_segment_embeddings = torch.zeros(decoder_input_encoding.size(0), decoder_input_encoding.size(1),
                                                 self.dim_vae_latent).to(vae_latent_space.device)

        if verbose:
            print("Decoder segment embeddings", decoder_segment_embeddings.shape)

        for n in range(decoder_input_encoding.size(1)):
            for b, (st, ed) in enumerate(zip(decoder_x_bar_position[n, :-1], decoder_x_bar_position[n, 1:])):
                decoder_segment_embeddings[st:ed, n, :] = vae_latent_space_reshaped[n, b, :]


        # Use for further conditioning in generation
        # Concatenate to the conditional embedding valence conditioning, if provided
        if valence_cls is not None:
            decoder_valence_embedding = self.valence_embedding(valence_cls)
            decoder_segment_embedding_cat = torch.cat([decoder_segment_embeddings, decoder_valence_embedding], dim=-1)
        else:
            decoder_segment_embedding_cat = decoder_segment_embeddings

        decoder_out = self.decoder(decoder_input_encoding, decoder_segment_embedding_cat)

        if verbose:
            print("Decoder output", decoder_out.shape)

        # Project to predict which is the most probable token
        decoder_logits = self.decoder_out_projection(decoder_out)

        if verbose:
            print("Decoder logits", decoder_logits.shape)

        return mu, logvar, decoder_logits

    def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
        recons_loss = F.cross_entropy(dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1),
                                      ignore_index=self.num_embedding - 1, reduction='mean').float()

        kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
        kldiv_loss = kl_after_free_bits.mean()

        return {
            'beta': beta,
            'total_loss': recons_loss + beta * kldiv_loss,
            'kldiv_loss': kldiv_loss,
            'kldiv_raw': kl_before_free_bits,
            'recons_loss': recons_loss
        }
