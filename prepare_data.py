import os
import sys
import pandas as pd
import re
import unicodedata
from collections import Counter
import json
import gc  # Ajout pour la gestion mémoire
import numpy as np  # Import numpy pour la gestion des types
from sklearn.model_selection import train_test_split
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)
import torch
import torchaudio
from tqdm import tqdm

# --- ============================== ---
# --- CONFIGURATIONS ET CHEMINS ---
# --- ============================== ---
PROJECT_ROOT = os.getcwd()
BASE_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
TARGET_SAMPLING_RATE = 16000
TEST_SET_SIZE = 0.15
VALIDATION_SET_SIZE = 0.10

# PARAMÈTRES OPTIMISÉS POUR LA MÉMOIRE
BATCH_SIZE_FOR_PROCESSING = 8  # Réduire si vous avez encore des problèmes de mémoire
MAX_AUDIO_LENGTH_SECONDS = 20  # Ignorer les audios trop longs
SAVE_INTERMEDIATE_BATCHES = True  # Sauvegarder par petits lots

FON_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "Fongbe_Speech_Dataset")
FON_METADATA_CSV = os.path.join(FON_DATASET_DIR, "fongbe_speech_dataset_metadata.csv")
FON_AUDIO_DIR = os.path.join(FON_DATASET_DIR, "fongbe_speech_audio_files", "wav")
FON_LAB_DIR = os.path.join(FON_DATASET_DIR, "fongbe_speech_audio_files", "lab")
YORUBA_CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "yorubaspeechcorpus", "corpus")

DATA_ETUDE_DIR = os.path.join(PROJECT_ROOT, "data_etude")
DATA_FINETUNING_DIR = os.path.join(PROJECT_ROOT, "data_finetuning")

# --- ============================== ---
# --- FONCTIONS DE NETTOYAGE ET PARSING (RENFORCÉES) ---
# --- ============================== ---
def clean_text_with_diacritics(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    chars_to_remove = '.,?!;:()"“%‘”'
    text = re.sub(f"[{re.escape(chars_to_remove)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_without_diacritics(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    chars_to_remove = '.,?!;:()"“%‘”'
    text = re.sub(f"[{re.escape(chars_to_remove)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_transcription(text):
    """Vérifie si une transcription est valide"""
    if not isinstance(text, str):
        return False
    if pd.isna(text) or pd.isnull(text):
        return False
    if not text.strip():
        return False
    if text.strip() in ['', 'nan', 'NaN', 'None', 'null']:
        return False
    return True

def get_audio_duration(audio_path):
    """Obtient la durée d'un fichier audio sans le charger complètement"""
    try:
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except:
        return 0

def parse_fongbe_speech_dataset(metadata_csv_path, base_audio_dir, base_lab_dir, cleaning_function):
    print(f"Parsing du dataset Fon depuis '{metadata_csv_path}'...")
    if not os.path.exists(metadata_csv_path):
        print(f"  ATTENTION: Fichier metadata '{metadata_csv_path}' non trouvé. Skip."); return pd.DataFrame()
    
    df_meta = pd.read_csv(metadata_csv_path)
    records = []
    
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="  Processing Fon CSV"):
        audio_filename = row.get('audio filename')
        transcription_filename = row.get('transcription filename')
        
        if pd.isna(audio_filename) or pd.isna(transcription_filename): 
            continue
        
        audio_path = os.path.join(base_audio_dir, audio_filename)
        transcription_path = os.path.join(base_lab_dir, transcription_filename)

        if os.path.exists(audio_path) and os.path.exists(transcription_path):
            # Vérifier la durée de l'audio avant de traiter
            duration = get_audio_duration(audio_path)
            if duration > MAX_AUDIO_LENGTH_SECONDS or duration == 0:
                print(f"  SKIP: Audio trop long ({duration:.2f}s) ou invalide: {audio_filename}")
                continue
                
            try:
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcription_text = f.read().strip()
                
                if is_valid_transcription(transcription_text):
                    cleaned_text = cleaning_function(transcription_text)
                    if is_valid_transcription(cleaned_text):
                        records.append({
                            "audio_path": os.path.abspath(audio_path),
                            "transcription": cleaned_text,
                            "language": "fon",
                            "duration": duration
                        })
                    else:
                        print(f"  SKIP: Transcription invalide après nettoyage pour {audio_filename}")
                else:
                    print(f"  SKIP: Transcription invalide dans {transcription_filename}")
                    
            except Exception as e:
                print(f"  ATTENTION: Impossible de lire '{transcription_path}'. Erreur: {e}")
    
    df = pd.DataFrame(records)
    print(f"  Trouvé {len(df)} échantillons Fon valides.")
    return df

def parse_yoruba_speech_corpus(corpus_dir, cleaning_function):
    print(f"Parsing du Yoruba Speech Corpus depuis '{corpus_dir}'...")
    records = []
    utts_files = [os.path.join(root, "utts.data") for root, _, files in os.walk(corpus_dir) if "utts.data" in files]
    
    for utts_file in tqdm(utts_files, desc="  Processing Yoruba Speakers"):
        speaker_path = os.path.dirname(utts_file)
        try:
            with open(utts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.search(r'\(\s*(\S+)\s+"(.+)"\s*\)', line)
                    if match:
                        utt_id, transcription = match.groups()
                        wav_path = os.path.join(speaker_path, "recordings", f"{utt_id}.wav")
                        
                        if os.path.exists(wav_path):
                            # Vérifier la durée de l'audio avant de traiter
                            duration = get_audio_duration(wav_path)
                            if duration > MAX_AUDIO_LENGTH_SECONDS or duration == 0:
                                print(f"  SKIP: Audio trop long ({duration:.2f}s) ou invalide: {utt_id}")
                                continue
                                
                            if is_valid_transcription(transcription):
                                cleaned_text = cleaning_function(transcription)
                                if is_valid_transcription(cleaned_text):
                                    records.append({
                                        "audio_path": os.path.abspath(wav_path),
                                        "transcription": cleaned_text,
                                        "language": "yoruba",
                                        "duration": duration
                                    })
                                else:
                                    print(f"  SKIP: Transcription Yoruba invalide après nettoyage pour {utt_id}")
                            else:
                                print(f"  SKIP: Transcription Yoruba invalide pour {utt_id}")
        except Exception as e:
            print(f"  ATTENTION: Erreur lors du parsing de {utts_file}: {e}")
    
    df = pd.DataFrame(records)
    print(f"  Trouvé {len(df)} échantillons Yoruba valides.")
    return df

def prepare_dataset_map_fn_single(example, processor):
    """
    Fonction de mapping optimisée pour la mémoire
    """
    try:
        # --- VÉRIFICATION RENFORCÉE DE LA TRANSCRIPTION ---
        transcription_text = example.get('transcription')
        
        if not is_valid_transcription(transcription_text):
            return {"input_values": [], "labels": []}

        transcription_text = str(transcription_text).strip()
        if not transcription_text:
            return {"input_values": [], "labels": []}

        # --- AUDIO ---
        audio_path = example["audio_path"]
        if not os.path.exists(audio_path):
            return {"input_values": [], "labels": []}
            
        # Chargement audio avec gestion mémoire
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resampling si nécessaire
            if sr != TARGET_SAMPLING_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLING_RATE)
                waveform = resampler(waveform)
                
            if waveform.ndim > 1: 
                waveform = waveform[0, :]
            
            if waveform.numel() == 0:
                return {"input_values": [], "labels": []}
            
            # Conversion en numpy et libération de la mémoire torch
            audio_array = waveform.squeeze().numpy().astype(np.float32)  # Force float32
            del waveform  # Libérer la mémoire
            
        except Exception as audio_error:
            print(f"  SKIP: Erreur audio pour {audio_path}: {audio_error}")
            return {"input_values": [], "labels": []}
        
        # --- TRAITEMENT ---
        try:
            # Traitement de l'audio avec type de données cohérent
            input_values = processor(audio_array, sampling_rate=TARGET_SAMPLING_RATE).input_values[0]
            
            # Forcer le type float32 pour éviter les conflits PyArrow
            input_values = np.array(input_values, dtype=np.float32).tolist()
            
            # Traitement du texte - version ultra-robuste sans tensors
            labels = processor.tokenizer.encode(transcription_text, add_special_tokens=False)
            
            # Vérification que c'est bien une liste d'entiers
            if not isinstance(labels, list):
                labels = list(labels)
                
            # Conversion explicite en int64 pour éviter les conflits
            labels = [int(x) for x in labels]
            
            if not all(isinstance(x, int) for x in labels) or len(labels) == 0:
                print(f"  SKIP: Labels invalides pour {audio_path}")
                return {"input_values": [], "labels": []}
                
        except Exception as processing_error:
            print(f"  SKIP: Erreur traitement pour {audio_path}: {processing_error}")
            return {"input_values": [], "labels": []}
        
        return {"input_values": input_values, "labels": labels}
            
    except Exception as e:
        print(f"  SKIP: Erreur générale pour l'échantillon {example.get('audio_path', 'N/A')}: {e}")
        return {"input_values": [], "labels": []}

def save_dataset_in_batches(dataset, output_path, batch_size=1000):
    """Sauvegarde un dataset par petits lots pour éviter les problèmes de mémoire"""
    print(f"    Sauvegarde par lots de {batch_size} échantillons...")
    
    os.makedirs(output_path, exist_ok=True)
    
    total_size = len(dataset)
    for i in range(0, total_size, batch_size):
        batch_end = min(i + batch_size, total_size)
        batch_dataset = dataset.select(range(i, batch_end))
        batch_path = os.path.join(output_path, f"batch_{i//batch_size:04d}")
        batch_dataset.save_to_disk(batch_path)
        print(f"      Lot {i//batch_size + 1}/{(total_size + batch_size - 1)//batch_size} sauvegardé")
        
        # Nettoyer la mémoire
        del batch_dataset
        gc.collect()

# --- ============================== ---
# --- BOUCLE DE PRÉPARATION PRINCIPALE ---
# --- ============================== ---
for mode in ["avec_diacritiques", "sans_diacritiques"]:

    print(f"\n{'='*20}\n--- DÉBUT DU TRAITEMENT EN MODE : {mode.upper()} ---\n{'='*20}")

    # 1. Sélectionner la fonction de nettoyage et les dossiers de sortie
    if mode == "avec_diacritiques":
        cleaning_function = clean_text_with_diacritics
        output_etude_dir = os.path.join(DATA_ETUDE_DIR, "corpus_avec_diacritiques")
        output_finetuning_dir = os.path.join(DATA_FINETUNING_DIR, "corpus_avec_diacritiques")
    else:
        cleaning_function = clean_text_without_diacritics
        output_etude_dir = os.path.join(DATA_ETUDE_DIR, "corpus_sans_diacritiques")
        output_finetuning_dir = os.path.join(DATA_FINETUNING_DIR, "corpus_sans_diacritiques")

    os.makedirs(output_etude_dir, exist_ok=True)
    os.makedirs(output_finetuning_dir, exist_ok=True)
    print(f"Les sorties pour ce mode seront dans:\n  Étude: {output_etude_dir}\n  Finetuning: {output_finetuning_dir}")

    # 2. Charger et parser toutes les données avec la bonne fonction de nettoyage
    print("\n[Étape 1/5] Parsing des données sources...")
    df_fon_all = parse_fongbe_speech_dataset(FON_METADATA_CSV, FON_AUDIO_DIR, FON_LAB_DIR, cleaning_function)
    df_yoruba_all = parse_yoruba_speech_corpus(YORUBA_CORPUS_DIR, cleaning_function)

    # Filtrage supplémentaire avec validation renforcée
    print("  Filtrage final des transcriptions...")
    df_fon_all = df_fon_all[df_fon_all["transcription"].apply(is_valid_transcription)]
    df_yoruba_all = df_yoruba_all[df_yoruba_all["transcription"].apply(is_valid_transcription)]

    # Trier par durée pour optimiser le traitement par lots
    df_fon_all = df_fon_all.sort_values('duration').reset_index(drop=True)
    df_yoruba_all = df_yoruba_all.sort_values('duration').reset_index(drop=True)

    print(f"  Après filtrage - Fon: {len(df_fon_all)}, Yoruba: {len(df_yoruba_all)}")
    print(f"  Durée moyenne - Fon: {df_fon_all['duration'].mean():.2f}s, Yoruba: {df_yoruba_all['duration'].mean():.2f}s")

    # Vérification qu'on a assez de données
    if len(df_fon_all) == 0 or len(df_yoruba_all) == 0:
        print("  ERREUR: Pas assez de données valides après filtrage!")
        continue

    # 3. Mettre de côté les données pour data_etude (évaluation)
    print("\n[Étape 2/5] Création des ensembles de test pour 'data_etude'...")
    fon_for_finetuning, fon_test = train_test_split(df_fon_all, test_size=TEST_SET_SIZE, random_state=42)
    yoruba_for_finetuning, yoruba_test = train_test_split(df_yoruba_all, test_size=TEST_SET_SIZE, random_state=42)
    
    # Supprimer la colonne duration pour les CSV finaux
    fon_test_clean = fon_test.drop('duration', axis=1)
    yoruba_test_clean = yoruba_test.drop('duration', axis=1)
    fon_for_finetuning = fon_for_finetuning.drop('duration', axis=1)
    yoruba_for_finetuning = yoruba_for_finetuning.drop('duration', axis=1)
    
    fon_test_clean.to_csv(os.path.join(output_etude_dir, "test_fon.csv"), index=False)
    yoruba_test_clean.to_csv(os.path.join(output_etude_dir, "test_yoruba.csv"), index=False)
    print(f"  Fichiers de test sauvegardés. Fon: {len(fon_test_clean)}, Yoruba: {len(yoruba_test_clean)}.")

    # 4. Préparer le corpus de fine-tuning et le partitionner
    print("\n[Étape 3/5] Création et partitionnement du corpus de fine-tuning...")
    df_finetuning_combined = pd.concat([fon_for_finetuning, yoruba_for_finetuning]).sample(frac=1, random_state=42)
    
    finetuning_train, finetuning_val = train_test_split(
        df_finetuning_combined, test_size=VALIDATION_SET_SIZE, random_state=42, 
        stratify=df_finetuning_combined['language']
    )
    
    finetuning_train.to_csv(os.path.join(output_finetuning_dir, "train.csv"), index=False)
    finetuning_val.to_csv(os.path.join(output_finetuning_dir, "validation.csv"), index=False)
    print(f"  Fichiers de fine-tuning sauvegardés. Train: {len(finetuning_train)}, Validation: {len(finetuning_val)}.")

    # 5. Créer les assets pour le fine-tuning (vocab, processor, hf_datasets)
    print("\n[Étape 4/5] Création des assets pour le fine-tuning...")
    assets_dir = os.path.join(output_finetuning_dir, "assets")
    vocab_dir = os.path.join(assets_dir, "vocab")
    processor_dir = os.path.join(assets_dir, "processor")
    hf_datasets_dir = os.path.join(assets_dir, "hf_datasets")
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(processor_dir, exist_ok=True)
    os.makedirs(hf_datasets_dir, exist_ok=True)

    # Créer le vocabulaire avec validation des transcriptions
    print("    Création du vocabulaire...")
    valid_transcriptions = [text for text in finetuning_train["transcription"].astype(str).tolist() if is_valid_transcription(text)]
    all_train_text = " ".join(valid_transcriptions)
    
    if not all_train_text.strip():
        print("    ERREUR: Aucun texte valide pour créer le vocabulaire!")
        continue
        
    vocab_dict = {char: i for i, char in enumerate(sorted(list(Counter(all_train_text).keys())))}
    with open(os.path.join(vocab_dir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"    Vocabulaire créé avec {len(vocab_dict)} tokens.")

    # Créer le processeur
    print("    Création du processeur...")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(vocab_dir, unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(processor_dir)
    print(f"    Processeur sauvegardé.")

    # Libérer la mémoire
    del valid_transcriptions, all_train_text
    gc.collect()

    # Créer et sauvegarder les datasets Arrow avec optimisations mémoire
    print("\n[Étape 5/5] Préparation et sauvegarde des datasets Hugging Face...")
    
    # Définir le schéma explicite pour éviter les conflits de types
    dataset_features = Features({
        'input_values': Sequence(Value('float32')),
        'labels': Sequence(Value('int64'))
    })
    
    # Traitement du dataset d'entraînement par petits lots
    print("    Traitement du dataset d'entraînement...")
    train_dataset_hf = Dataset.from_pandas(finetuning_train)
    
    # Libérer la mémoire du DataFrame
    del finetuning_train
    gc.collect()
    
    train_dataset_processed = train_dataset_hf.map(
        prepare_dataset_map_fn_single, 
        fn_kwargs={"processor": processor}, 
        remove_columns=list(train_dataset_hf.column_names),
        num_proc=1,
        batch_size=BATCH_SIZE_FOR_PROCESSING,
        features=dataset_features  # Spécifier le schéma
    )
    
    # Libérer la mémoire
    del train_dataset_hf
    gc.collect()

    print("    Traitement du dataset de validation...")
    val_dataset_hf = Dataset.from_pandas(finetuning_val)
    
    # Libérer la mémoire du DataFrame
    del finetuning_val
    gc.collect()
    
    val_dataset_processed = val_dataset_hf.map(
        prepare_dataset_map_fn_single,
        fn_kwargs={"processor": processor},
        remove_columns=list(val_dataset_hf.column_names),
        num_proc=1,
        batch_size=BATCH_SIZE_FOR_PROCESSING,
        features=dataset_features  # Spécifier le schéma
    )
    
    # Libérer la mémoire
    del val_dataset_hf
    gc.collect()

    # Filtrer les erreurs de mapping
    print("    Filtrage des échantillons invalides...")
    train_dataset_processed = train_dataset_processed.filter(lambda x: len(x["input_values"]) > 0 and len(x["labels"]) > 0)
    val_dataset_processed = val_dataset_processed.filter(lambda x: len(x["input_values"]) > 0 and len(x["labels"]) > 0)

    print(f"    Sauvegarde des datasets Arrow...")
    if SAVE_INTERMEDIATE_BATCHES and len(train_dataset_processed) > 2000:
        # Sauvegarder par lots pour les gros datasets
        save_dataset_in_batches(train_dataset_processed, os.path.join(hf_datasets_dir, "train"), batch_size=1000)
        save_dataset_in_batches(val_dataset_processed, os.path.join(hf_datasets_dir, "validation"), batch_size=500)
    else:
        # Sauvegarde normale pour les petits datasets
        train_dataset_processed.save_to_disk(os.path.join(hf_datasets_dir, "train"))
        val_dataset_processed.save_to_disk(os.path.join(hf_datasets_dir, "validation"))
    
    print(f"    Datasets Arrow sauvegardés. Train: {len(train_dataset_processed)}, Validation: {len(val_dataset_processed)}.")
    
    # Libérer la mémoire à la fin de chaque mode
    del train_dataset_processed, val_dataset_processed, processor
    gc.collect()
    
    # Forcer le nettoyage de la mémoire GPU si disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*20}\n--- TRAITEMENT TERMINÉ ---\n{'='*20}")