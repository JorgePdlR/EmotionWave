import Model.EmotionWave as emw
import torch
from MIDIoperations import REMItokenizer, MidiWav


def main():
    emw_model = emw.EmotionWave(12, 512, 8, 2048,
                    128, 512, 10000, 12, 512,
                    8, 2048)
    #print(emw_model)

    # Define the shapes
    seqlen_per_bar = 50    # Sequence length per bar
    bsize = 2              # Batch size
    n_bars_per_sample = 8  # Number of bars per sample
    seqlen_per_sample = seqlen_per_bar * n_bars_per_sample

    # Create synthetic inputs with appropriate types
    enc_inp = torch.randint(0, 100, (seqlen_per_bar, bsize, n_bars_per_sample), dtype=torch.long)
    dec_inp = torch.randint(0, 100, (seqlen_per_sample, bsize), dtype=torch.long)
    dec_inp_bar_pos = torch.randint(0, n_bars_per_sample + 1, (bsize, n_bars_per_sample + 1), dtype=torch.long)
    dec_tgt = torch.randint(0, 10000, (seqlen_per_sample, bsize), dtype=torch.long)
    valence_cls = torch.randint(0, 8, (seqlen_per_sample, bsize), dtype=torch.long)

    mu, logvar, decoder_logits = emw_model(enc_inp, dec_inp, dec_inp_bar_pos, valence_cls=valence_cls, verbose=True)
    print(decoder_logits.shape)
    vloss = emw_model.compute_loss(mu, logvar, 1.0, 0.25, decoder_logits, dec_tgt)
    print(vloss)
    print("DONE")


main()
