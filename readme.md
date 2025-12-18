# Multilingual ASR for African Languages (XLS-R)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the source code for the Master's Thesis: **"Conception d'un modèle multilingue pour les langues africaines"**.

It implements a fine-tuning pipeline for **Facebook's XLS-R (Wav2Vec2)** model specifically adapted for low-resource African languages, with a primary focus on **Fongbe** and **Yoruba**.

## 📂 Project Structure
├── data/

│ ├── fon_corpus/ # Raw audio and transcripts for Fongbe

│ └── yoruba_corpus/ # Raw audio and transcripts for Yoruba

├── finetuning/

│ ├── prepare_data.py # Data preprocessing, cleaning, and vocab generation

│ └── run_finetuning_v2.py # Main training loop using Hugging Face Trainer

├── requirements.txt # Python dependencies

└── README.md
code
Code
## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/charbossly/xlsr-wav2vec2-multilingual-asr-african-languages.git
   cd xlsr-wav2vec2-multilingual-asr-african-languages
   
### Create a virtual environment (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install dependencies:
    ```bash
    pip install -r requirements.txt
    
(Note: Ensure you have torch, transformers, datasets, librosa, jiwer, and soundfile installed).

**Reproduction (How to Run)**

To reproduce the results obtained in the thesis (Fine-tuning on Fongbe + Yoruba), follow these steps:

1. Data Preparation
The prepare_data.py script handles audio resampling (16kHz), text cleaning (normalization, diacritics handling), and creates the training/validation splits.
      ```bash
      python finetuning/prepare_data.py
      Output: Processed datasets saved in Arrow format and a vocab.json file.

2. Fine-Tuning
The run_finetuning_v2.py script loads the pre-trained facebook/wav2vec2-large-xlsr-53 model and fine-tunes it on the processed data.
    ```bash
    python finetuning/run_finetuning_v2.py \
        --output_dir="./xlsr-finetuned-fon-yor" \
        --num_train_epochs 30 \
        --per_device_train_batch_size 8 \
        --learning_rate 3e-4 \
        --fp16
       
🌍 Extension: How to Add a New Language

This pipeline was designed to be modular. Follow this roadmap to add a new language (e.g., Goun, Mina, Wolof) to the model.


**Step 1: Prepare the Data**
Create a new folder in data/ (e.g., data/goun_corpus/). Ensure your data follows one of these structures:
CSV Format: path/to/audio.wav, transcript
Kaldi Format: text file and wav.scp file.

**Step 2: Update the Preprocessing Script**
Modify finetuning/prepare_data.py to include the new language path.
    ```bash
    # In prepare_data.py
    
    def load_all_datasets():
        fon_data = load_dataset(...) 
        yoruba_data = load_dataset(...)
        
        # 1. Add your new language here
        goun_data = load_dataset("csv", data_files="data/goun_corpus/metadata.csv")
        
        # 2. Concatenate with existing data
        multilingual_dataset = concatenate_datasets([fon_data, yoruba_data, goun_data])
        
        return multilingual_dataset
        
**Step 3: Update Vocabulary**
The script automatically builds the vocabulary from the training data. However, if your new language introduces specific characters (unique diacritics not present in Fon/Yoruba), ensure they are not removed by the cleaning regex function in prepare_data.py.

**Step 4: Incremental Training (Optional)**
If you want to keep the knowledge from the existing Fon/Yoruba model, load the locally saved checkpoint instead of the Facebook base model in run_finetuning_v2.py:
   ```bash  # Change this line:
    model_name = "facebook/wav2vec2-large-xlsr-53"
    # To this:
    model_name = "./xlsr-finetuned-fon-yor/checkpoint-best"
    ```

Then run the training script again.
## 📊 Results
Language	Model	WER (%)	CER (%)
Fongbe	Base XLS-R	100%	100%
Fongbe	Fine-Tuned	29.93%	9.96%
Yoruba	Base XLS-R	100%	92.88%
Yoruba	Fine-Tuned	51.15%	13.95%


### Author
Mahoudjro Charbel ASSOGBA
Master in Software Engineering

## Contributing
Contributions are welcome! If you have datasets for other African languages, please open an issue or submit a pull request.
