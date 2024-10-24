import numpy as np
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from torch.utils.data import DataLoader, Dataset

# Load processed audio data and transcripts
processed_audios = np.load('processed_audios.npy', allow_pickle=True)
transcripts = np.load('transcripts.npy', allow_pickle=True)

# Parameters
target_length = 16000  # Adjust according to your needs

# Function to preprocess audio data
def preprocess_audios(audios, target_length):
    processed = []
    for audio in audios:
        if len(audio) < target_length:
            padded_audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            processed.append(padded_audio)
        else:
            processed.append(audio[:target_length])
    return np.array(processed)

# Preprocess audio clips
audio_tensors = preprocess_audios(processed_audios, target_length)
audio_tensors = torch.tensor(audio_tensors, dtype=torch.float32)

# Define the dataset class
class AudioTextDataset(Dataset):
    def __init__(self, audio_tensors, transcripts):
        self.audio_tensors = audio_tensors
        self.transcripts = transcripts

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        return {
            'input_values': self.audio_tensors[idx],  # Audio tensor
            'text': str(self.transcripts[idx])          # Ensure transcript is a string
        }

# Create a dataset and DataLoader
dataset = AudioTextDataset(audio_tensors, transcripts)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load SpeechT5 model and processor
model_name = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(model_name)
model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Fine-tuning loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        # Get the text input
        text_inputs = batch['text']

        # Ensure text_inputs are standard Python strings
        text_inputs = [str(text) for text in text_inputs]

        # Debugging: Print the contents of the batch
        print("Text inputs:", text_inputs)

        # Prepare text inputs for the processor
        text_inputs = processor(text=text_inputs, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)

        # Forward pass (pass tokenized text as input, no audio)
        outputs = model(input_ids=text_inputs.long(), labels=text_inputs.long())
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_speacht5")
processor.save_pretrained("fine_tuned_speacht5")

print("Fine-tuning completed and model saved.")
