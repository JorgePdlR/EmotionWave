# Using REMI scheme tokenisation
from miditok import REMI, TokenizerConfig
# Manage paths
from pathlib import Path
import os
# Librosa for music and audio analysis
import librosa
# Symbolic music datasets library
import muspy
# To calculate information from the midi file
import music21
# To copy objects
import copy
# Ipython.display to show audio files
import IPython.display as ipd
# To convert MIDI to wav
import pretty_midi
from midi2audio import FluidSynth

class MidiWav:
    def __init__(self, file_path):
        # File exists, store path
        if os.path.exists(file_path):
            self.midi_path = file_path
            self.wav_path = ""
        # File doesn't exist, signal error
        else:
            raise FileNotFoundError(f"No file found at specified path: {file_path}")

    def convert_to_wav(self, audio_path, sound_font='default.sf2'):

        if not os.path.isfile(sound_font):
            raise FileNotFoundError(f"Sound font not found: {sound_font}")
        # Load the MIDI file
        midi_data = pretty_midi.PrettyMIDI(self.midi_path)

        # Save MIDI data to a temporary MIDI file
        temp_midi_path = 'temp_output.mid'
        midi_data.write(temp_midi_path)

        # Use FluidSynth to convert the MIDI file to WAV using the specified sound font
        fs = FluidSynth(sound_font)
        fs.midi_to_audio(temp_midi_path, audio_path)

        self.wav_path = audio_path

    def reproduce_audio(self):
        # Convert .mid to .wav if not done yet. If done at this point
        # it will save the file in the current directory with the original
        # name but with the extension .wav
        if self.wav_path == "":
            # Extract file name
            file_name = os.path.basename(self.midi_path)
            # Remove .mid termination and add .wav termination
            file_name = file_name[0:-4] + ".wav"
            self.convert_to_wav(file_name)

        # Load the audio file, using native sample rate
        audio_data, sample_rate = librosa.load(self.wav_path, sr=None)

        # Reproduce the audio
        ipd.display(ipd.Audio(audio_data, rate=sample_rate))

    def get_midi_duration_in_seconds(self):
        mf = music21.midi.MidiFile()
        # Open, read and close MIDI file
        mf.open(self.midi_path)
        mf.read()
        mf.close()

        score = music21.midi.translate.midiFileToStream(mf)

        # Find all tempo changes and their offsets
        tempos = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
        tempo_changes = [(tempo.offset, tempo.number) for tempo in tempos]

        # Default to 120 BPM if no tempo marking is found
        if not tempo_changes:
            tempo_changes = [(0, 120)]

        # Add an end point for the last section
        total_quarter_length = score.duration.quarterLength
        tempo_changes.append((total_quarter_length, tempo_changes[-1][1]))

        # Calculate duration in seconds
        duration_in_seconds = 0
        for i in range(len(tempo_changes) - 1):
            start_offset, bpm = tempo_changes[i]
            end_offset = tempo_changes[i + 1][0]
            quarter_length_duration = end_offset - start_offset
            seconds_per_beat = 60 / bpm
            duration_in_seconds += quarter_length_duration * seconds_per_beat

        return duration_in_seconds

    def truncate_midi(self, start_second=0, duration=30, id=0, update_path=False,
                      dir_path='', verbose=0):
        # Load the MIDI file; according to the documentation of music21:
        #  "By default, MIDI streams are quantized to the nearest sixteenth or
        #   triplet-eighth (i.e. smaller durations will not be preserved)."
        # Hence, setting quantizePost as False to avoid quantisation
        score = music21.converter.parse(self.midi_path, quantizePost=False)

        # Create a new stream to store the truncated parts
        truncated_score = music21.stream.Score()

        # Calculate the start and end times in quarter lengths
        metronome_score = score.metronomeMarkBoundaries()

        metronome_sstart = metronome_score[0][0]

        # Reverse the list to start from the end. This is done
        # in this way because in several songs metronome marks are redundant
        # and the last metronome mark of the same offset, is the one that
        # applies to the following notes
        metronome_score.reverse()

        ssmetronome = None
        # Get metronome mark for the corresponding time frame
        for metronome in metronome_score:
            if metronome_sstart == metronome[0]:
                ssmetronome = metronome
                break

        # Assume that the maximum metronome score is the correct one for all the
        # song. This will allow us to get at least the total amount of notes in the
        # range always.
        if ssmetronome:
            bpm = ssmetronome[2].number
        else:
            # Default bits per minute
            bpm = 120

        seconds_per_beat = 60 / bpm
        start_quarter_length = start_second / seconds_per_beat
        end_quarter_length = (start_second + duration) / seconds_per_beat

        # We are assuming all scores will have parts, Check if the score has
        # no parts and assert if that is the case.
        if not score.parts:
            raise AssertionError("No parts found in the score.")

        # Process each part separately
        for part in score.parts:
            truncated_part = music21.stream.Part()
            nmetronome = None
            ninstrument = None
            ntime = None
            nsigk = None
            # Used to keep track of how much seconds have been converted
            seconds_so_far = 0

            # Get all the metronome marks with boundaries in a list
            metronome_part = part.metronomeMarkBoundaries()
            # Reverse the list to start from the end of the song
            metronome_part.reverse()

            # Get metronome mark for the corresponding time frame
            for metronome in metronome_part:
                if metronome[0] <= start_quarter_length <= metronome[1]:
                    nmetronome = metronome[2]
                    break

            # Set information just if first measure is not coppied
            if start_quarter_length != 0:
                # Get instrument of this part
                ninstrument = truncated_part.getInstrument()

                # Get time signature
                tsignatures = truncated_part.getTimeSignatures(recurse=False)
                ntime = tsignatures[0]

                # Get key signature
                nsigk = part.flatten().getElementBeforeOffset(1, music21.key.KeySignature)

                # Set instrument
                if ninstrument:
                    truncated_part.append(copy.deepcopy(ninstrument))

                # Set metronome
                if nmetronome:
                    truncated_part.append(copy.deepcopy(nmetronome))

                # Set time signature
                if ntime:
                    truncated_part.append(copy.deepcopy(ntime))

                # Set key signature
                if nsigk:
                    truncated_part.append(copy.deepcopy(nsigk))

            # Get all elements (notes and extra information) within the range
            truncated_elements = part.getElementsByOffset(
                offsetStart=start_quarter_length,
                offsetEnd=end_quarter_length
            )

            # Adjust offsets and add elements to the new part
            for element in truncated_elements:
                new_element = copy.deepcopy(element)
                new_element.offset -= start_quarter_length
                # Appending here to add each element to the end of the part
                truncated_part.append(new_element)

                # Just if this is a measure
                if isinstance(element, music21.stream.Measure):
                    # Calculate the time spent in this measure
                    metronome_measure = element.metronomeMarkBoundaries()

                    # Get the first metronome of this measure and use it to calculate the
                    # time spent
                    if metronome_measure:
                        metronome_start = metronome_measure[0][0]

                        # Reverse the list to start from the end of the measure. This is
                        # done in this way because in several songs metronome marks are
                        # redundant and the last metronome mark of the same offset, is the
                        # one that applies to the following notes
                        metronome_measure.reverse()

                        tmetronome = None
                        # Get metronome mark for the corresponding time frame
                        for metronome in metronome_measure:
                            if metronome_start == metronome[0]:
                                tmetronome = metronome
                                break

                        # Get the bmp
                        if tmetronome:
                            bpm = tmetronome[2].number
                            seconds_per_beat = 60 / bpm

                    seconds_so_far += seconds_per_beat * element.duration.quarterLength

                    if seconds_so_far > duration:
                        break

            # Make sure that all parts start at offset 0.
            truncated_part.offset = 0

            # Add part just if it has at least one measure
            if part.hasMeasures():
                # Insert to guarantee part is added at offset 0
                truncated_score.insert(truncated_part)

        # If there is no measure in this score, exit. Otherwise continue, create
        # midi and store it
        there_is_a_part = False
        for part in truncated_score.parts:
            if not part.hasMeasures():
                # Remove this part from the score, it is empty
                truncated_score.remove(part)
            else:
                # With at least one measure in one part we can create a score
                there_is_a_part = True

        # Return if there is at least one part
        if not there_is_a_part:
            return False

        if verbose == 2:
            print("New score printing\n")
            dump_stream_hierarchy(truncated_score, 0)
            print("Old score printing\n")
            dump_stream_hierarchy(score, 0)

        # Convert the new stream to a MIDI file and save it
        new_mf = music21.midi.translate.music21ObjectToMidiFile(truncated_score)

        # Extract file name
        file_name = os.path.basename(self.midi_path)
        # Save the new MIDI file
        truncated_midi_name = f"{file_name[:-4]}_{id}.mid"
        truncated_midi_path = dir_path + truncated_midi_name
        new_mf.open(truncated_midi_path, 'wb')
        new_mf.write()
        new_mf.close()

        if verbose == 1:
            print(truncated_midi_path)

        # Update the path to the truncated MIDI
        if update_path:
            self.midi_path = truncated_midi_path

        return True


class REMItokenizer():
    def __init__(self, parameters):
        self.configuration = TokenizerConfig(**parameters)
        self.tokenizer = REMI(self.configuration)

    def tokenize_midi_file(self, file_path):
        tokens = self.tokenizer(Path(file_path))
        return tokens

    def split_per_bars(self, tokenized_song):
        tracks = []
        # Go through all the tracks in the song
        for track in tokenized_song:
            # Store a list of bars per instrument. The structure is:
            # list[track1[bar1, bar2, ..., barN], track2[bar1, bar2, ..., barN]]
            tracks.append(track.split_per_bars())

        # Return None if the list is empty
        if not tracks:
            tracks = None

        return tracks

    def _split_with_n_tracks(self, song_in_bars, num_of_bars, num_of_track):
        divided_song = []
        # Get the total number of bars in the song
        total_bars = len(song_in_bars[0])
        # Counter of number of bars processed
        current_bars = 0
        # Get starting bar
        start = current_bars
        # Calculate ending bar for next iteration
        end = current_bars + num_of_bars

        # Divide the track in lists of bars multiple of num_of_bars
        while start < total_bars:
            song_fragment = []
            for track_num in range(num_of_track):
                track = song_in_bars[track_num]
                song_fragment.append(track[start: end])

            # Add total number of bars
            current_bars += num_of_bars
            # Get starting bar
            start = current_bars
            # Calculate ending bar for next iteration
            end = current_bars + num_of_bars
            # Add song_fragment to divided song list
            divided_song.append(song_fragment)

        return divided_song

    def split_in_groups_of_bars(self, tokenized_song, num_of_bars=8, any_num_of_tracks=False):
        # Split the song in tracks and bars
        song_in_bars = self.split_per_bars(tokenized_song)
        total_tracks = len(song_in_bars)
        divided_song = []

        # Assuming the song will be divided in two tracks, melody and harmony, or just 1 track. Return None if these
        # assumptions are not True.
        if (total_tracks > 2 or total_tracks == 0) and not any_num_of_tracks:
            print(f"Number of tracks is more than 2!: {len(song_in_bars)}")
            return None

        # Ignore songs with more than 2 tracks if any_num_of_tracks is False
        if not any_num_of_tracks:
            # Divide according to the case
            if total_tracks == 1:
                divided_song = self._split_with_n_tracks(song_in_bars, num_of_bars, 1)
            elif total_tracks == 2:
                divided_song = self._split_with_n_tracks(song_in_bars, num_of_bars, 2)
            else:
                print(f"Number of tracks is more than 2 or 0!: {len(song_in_bars)}")
                divided_song = None
        else:
            divided_song = self._split_with_n_tracks(song_in_bars, num_of_bars, total_tracks)

        return total_tracks, divided_song

    def tokens_to_midi(self, file_path, tokens):
        generated_midi = self.tokenizer(tokens)
        generated_midi.dump_midi(Path(file_path))

    def divide_songs_in_fragments_from_dataset(self, paths, dir_path, num_of_bars=8, any_num_of_tracks=False):
        min = 0  # Track number of fragments smaller than segment_size
        total_fragments = 0  # Track the total number of fragments created

        # If directory to store files does not exist, create it
        os.makedirs(dir_path, exist_ok=True)

        # Go through all the paths in the dataset and split each song in the corresponding bars and according to the
        # number of tracks
        for i, song in enumerate(paths):
            current = MidiWav(song)

            # Print progress every 100 songs
            if i % 100 == 0:
                print("Number of songs processed:", i, "total number of fragments so far:", total_fragments)
                print(min, "Songs smaller than 30 seconds so far")

            # Time to create the fragments
            tokenized_song = self.tokenize_midi_file(current.midi_path)
            _, divided_song = self.split_in_groups_of_bars(tokenized_song, num_of_bars=num_of_bars,
                                                           any_num_of_tracks=any_num_of_tracks)

            # Go through all fragments and store them
            for fragment_id, fragment in divided_song:
                # Create name for the fragment
                # Extract file name
                file_name = os.path.basename(current.midi_path)
                # Save the new MIDI file
                truncated_midi_name = f"{file_name[:-4]}_{fragment_id}.mid"
                truncated_midi_path = dir_path + truncated_midi_name

                # Convert tokens to midi and store it
                self.tokens_to_midi(truncated_midi_path, fragment)

                # Count each fragment
                total_fragments += 1

                # Get metrics of the number of songs that are below the num_of_bars
                if len(fragment) < num_of_bars:
                    min += 1

        # Return metrics, total number of created fragments, and number of fragments
        # smaller than the number of bars
        return total_fragments, min


#### Debugging functions ####
def dump_stream_hierarchy(stream, level=0):
    indent = "  " * level
    for element in stream:
        if isinstance(element, music21.stream.Stream):
            print(
                f"{indent}{element.__class__.__name__} (offset: {element.offset}, duration: {element.duration.quarterLength})")
            dump_stream_hierarchy(element, level + 1)
        elif isinstance(element, music21.note.Note):
            tie_info = f", tie: {element.tie.type}" if element.tie else ""
            velocity_info = f", velocity: {element.volume.velocity}" if element.volume.velocity else ""
            lyric_info = f", lyric: {element.lyric}" if element.lyric else ""
            print(
                f"{indent}Note (pitch: {element.pitch}, offset: {element.offset}, duration: {element.duration.quarterLength}{tie_info}{velocity_info}{lyric_info})")
        elif isinstance(element, music21.chord.Chord):
            pitches = ', '.join(str(p) for p in element.pitches)
            root_info = f", root: {element.root}" if element.root else ""
            bass_info = f", bass: {element.bass}" if element.bass else ""
            velocity_info = f", velocity: {element.volume.velocity}" if element.volume.velocity else ""
            closed_position = ', '.join(str(p) for p in element.closedPosition().pitches)
            print(
                f"{indent}Chord (pitches: {pitches}, offset: {element.offset}, duration: {element.duration.quarterLength}{velocity_info}{root_info}{bass_info})")
            print(f"{indent}  Closed Position: {closed_position}")
        elif isinstance(element, music21.instrument.Instrument):
            print(f"{indent}Instrument: {element.instrumentName}")
        elif isinstance(element, music21.clef.Clef):
            print(f"{indent}Clef: {element.sign}{element.line}")
        elif isinstance(element, music21.meter.TimeSignature):
            print(f"{indent}TimeSignature: {element.ratioString}")
        elif isinstance(element, music21.key.KeySignature):
            print(f"{indent}KeySignature: {element.sharps} sharps/flats")
        elif isinstance(element, music21.tempo.MetronomeMark):
            number_info = f", number (BPM): {element.number}" if element.number else ""
            text_info = f", text: {element.text}" if element.text else ""
            print(
                f"{indent}MetronomeMark (offset: {element.offset}, duration: {element.duration.quarterLength}{number_info}{text_info})")
        else:
            print(
                f"{indent}{element.__class__.__name__} (offset: {element.offset}, duration: {element.duration.quarterLength})")
