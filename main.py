import Model.EmotionWave as emw
from MIDIoperations import REMItokenizer, MidiWav


def main():
    #module = emw.EmotionWave(12, 512, 8, 2048,
    #                128, 512, 10000, 12, 512,
    #                8, 2048)
    #print(module)
    #print("DONE")
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["BOS", "EOS", "MASK"],
        "use_chords": True,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
    }
    remi = REMItokenizer(TOKENIZER_PARAMS)
    current = MidiWav(r"C:\Users\jored\Downloads\Animal Crossing_Nintendo 3DS_Animal Crossing New Leaf_100 AM_0.mid")
    # current.convert_to_wav("personal_tests.wav", r"C:\Users\jored\GeneralUser_GS_1.471\GeneralUser GS 1.471\GeneralUser GS v1.471.sf2")
    tokenized_song = remi.tokenize_midi_file(current.midi_path)
    _, divided_song = remi.split_in_groups_of_bars(tokenized_song, num_of_bars=8)
    for fragment in divided_song:
        generated_midi = remi.tokens_to_midi("new_mid.mid", fragment)
        print(fragment)


main()
