# Fine-Tuning-SpeechT5-on-English-Technical-Jargon

This project provides the setup and code to fine-tune the SpeechT5 model for accurately handling English technical jargon (e.g., API, CUDA, TTS). The guide walks you through dataset preparation, audio preprocessing, and fine-tuning.

Prerequisites
Ensure you have the following installed on your system:

Python 3.10+

GPU (optional but recommended for faster training)

Git

Installation
Follow these steps to set up your environment and install dependencies.

Step 1: Clone the Repository
First, clone this repository to your local machine:

bash
Copy code

      git clone https://github.com/your-repo/fine-tune-speechT5-technical-jargon.git
      cd fine-tune-speechT5-technical-jargon

Step 2: Set Up a Virtual Environment (Optional)
It’s recommended to use a virtual environment to isolate dependencies:

bash
Copy code
      python -m venv venv
      source venv/bin/activate  # On Linux/macOS
      
 OR
 
      venv\Scripts\activate      # On Windows
Step 3: Install Requirements
The required Python packages are listed in the requirements.txt file. Install them as follows:

bash
Copy code

       pip install -r requirements.txt
      
After installing the dependencies, verify with the following command:

bash
Copy code

        python -c "import torch, librosa, pandas, transformers"
