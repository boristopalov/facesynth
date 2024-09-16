import itertools

from ableton import send_osc_message


midi_start_time = 0


# Define the chromatic scale
CHROMATIC_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Define scale patterns (intervals from the root note)
SCALE_PATTERNS = {
    'angry': [0, 2, 3, 5, 6, 8, 9, 11],  # Diminished scale
    'disgust': [0, 1, 3, 4, 6, 7, 9, 10],  # Half-Whole Diminished scale
    'fear': [0, 1, 3, 5, 7, 8, 10],  # Phrygian scale
    'happy': [0, 2, 4, 6, 7, 9, 11],  # Lydian scale
    'sad': [0, 2, 3, 5, 7, 8, 10],  # Natural Minor scale
    'surprise': [0, 2, 4, 6, 8, 10],  # Whole Tone scale
    'neutral': [0, 2, 4, 5, 7, 9, 11]  # Major scale
}


# TODO: understand
# def generate_scale(root_note, pattern):
#     """Generate a scale starting from the root note using the given pattern."""
#     root_index = CHROMATIC_SCALE.index(root_note)
#     chromatic_cycle = itertools.cycle(CHROMATIC_SCALE[root_index:] + CHROMATIC_SCALE[:root_index])
#     return [next(chromatic_cycle) for _ in pattern]

def generate_scale(root_note, pattern):
    """Generate a scale starting from the root note using the given pattern."""
    root_index = CHROMATIC_SCALE.index(root_note)
    return [CHROMATIC_SCALE[(root_index + interval) % 12] for interval in pattern]



def get_emotion_scales(root_note):
    """Generate scales for all emotions starting from the given root note."""
    return {emotion: generate_scale(root_note, pattern) 
            for emotion, pattern in SCALE_PATTERNS.items()}

def note_to_midi(note):
    """Convert a note name to its MIDI note number."""
    octave = int(note[-1]) if note[-1].isdigit() else 3  # Default to octave 3 if not specified
    note_name = note[:-1] if note[-1].isdigit() else note
    note_index = CHROMATIC_SCALE.index(note_name)
    return (octave + 1) * 12 + note_index

def scale_to_midi(scale, start_octave=3):
    """Convert a scale (list of note names) to MIDI note numbers."""
    midi_numbers = []
    for i, note in enumerate(scale):
        octave = start_octave + (i // 12)  # Increase octave every 12 notes
        midi_number = note_to_midi(f"{note}{octave}")
        midi_numbers.append(midi_number)
    return midi_numbers

def send_midi_notes_to_ableton(track_id, clip_id, midi_notes, duration=1, velocity=64):
    global midi_start_time
    """
    Send MIDI notes to Ableton Live using AbletonOSC.
    
    :param track_id: The ID of the track in Ableton Live
    :param clip_id: The ID of the clip in Ableton Live
    :param midi_notes: List of MIDI note numbers
    :param start_time: Start time of the first note (in beats)
    :param duration: Duration of each note (in beats)
    :param velocity: Velocity of the notes (0-127)
    """
    command = "/live/clip/add/notes"
    
    # Prepare the parameters for each note
    note_params = []
    for i, pitch in enumerate(midi_notes):
        # note_start_time = start_time + i * duration  # Each note starts after the previous one
        midi_start_time += duration # Each note starts after the previous onen
        note_params.extend([pitch, midi_start_time, duration, velocity, False])
    
    # Prepare the full parameter list
    params = [track_id, clip_id] + note_params
    
    # Send the message
    result = send_osc_message(command, params)
    return result

if __name__ == "__main__":
    root_note = 'C'
    emotion_scales = get_emotion_scales(root_note)
    
    for emotion, scale in emotion_scales.items():
        midi_scale = scale_to_midi(scale)
        print(f"{emotion.capitalize()} scale: {scale}")
        print(f"MIDI numbers: {midi_scale}")
        print()

    result = send_osc_message("/live/track/get/name", [0])
    result = send_osc_message("/live/clip/get/notes", [0, 0])

    # test_midi = [48, 50, 52, 54, 55, 57, 59]
    # resp = send_midi_notes_to_ableton(0, 0, test_midi)
    # print("response", resp)
    print(result)