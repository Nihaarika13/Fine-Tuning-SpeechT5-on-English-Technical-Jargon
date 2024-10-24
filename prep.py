import os
import numpy as np
import librosa
import pandas as pd


# Define constants
AUDIO_DIR = 'E:/speecht5/dataset/audio'  # Path to your audio files
TRANSCRIPT_DIR = 'E:/speecht5/dataset/transcriptions'  # Path to your transcripts
MAPPING_FILE = 'E:/speecht5/dataset/mapping.csv'  # Path to your mapping CSV
TARGET_LENGTH = 8 * 16000  # Target length for audio in samples (e.g., 8 seconds)

# Function to preprocess audio files
def preprocess_audio(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    # Convert to mono (if stereo)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    # Pad the audio with zeros to the target length
    if len(audio) < TARGET_LENGTH:
        audio = np.pad(audio, (0, max(0, TARGET_LENGTH - len(audio))), 'constant')
    else:
        audio = audio[:TARGET_LENGTH]  # Truncate if it's longer than target length
    return audio

# Read the mapping CSV
mapping_df = pd.read_csv(MAPPING_FILE)

# Prepare to store processed audio and transcripts
processed_audios = []
transcripts = []

# Iterate over the mapping DataFrame
for index, row in mapping_df.iterrows():
    audio_file = os.path.join(AUDIO_DIR, row['audio_file'])
    transcript_file = os.path.join(TRANSCRIPT_DIR, row['transcription_file'])

    # Preprocess the audio
    try:
        audio = preprocess_audio(audio_file)
        processed_audios.append(audio)

        # Read the transcript
        with open(transcript_file, 'r') as f:
            transcript = f.read().strip()
            transcripts.append(transcript)

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Convert processed audio list to a NumPy array
processed_audios = np.array(processed_audios)

# Save processed data for later use
np.save('processed_audios.npy', processed_audios)
np.save('transcripts.npy', transcripts)

print("Audio processing completed.")
