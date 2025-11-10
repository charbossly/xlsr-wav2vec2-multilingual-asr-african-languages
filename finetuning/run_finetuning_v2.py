import os
import sys
import numpy as np
import evaluate
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    DataCollatorForCTC
)
from datasets import load_from_disk, concatenate_datasets
import torch
import wandb
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# --- ============================== ---
# --- PARAMÈTRES DE L'EXPÉRIENCE (À MODIFIER PAR L'UTILISATEUR) ---
# --- ============================== ---

# Choisis le mode pour cette exécution : "avec_diacritiques" ou "sans_diacritiques"
MODE = "avec_diacritiques" 

# Définit le nombre TOTAL d'époques que tu veux atteindre à la fin de cette session.
# Si tu commences (0 époque faite) et que tu mets 5, il s'entraînera pour 5 époques.
# Si tu as déjà fait 5 époques et que tu mets 10, il s'entraînera pour 5 époques supplémentaires.
TOTAL_TRAIN_EPOCHS = 100

# --- ============================== ---
# --- CONFIGURATIONS ET CHEMINS (AUTOMATIQUES) ---
# --- ============================== ---
PROJECT_ROOT = os.path.dirname(os.getcwd()) # On est dans finetuning/, on remonte

# Chemins dynamiques basés sur le MODE
DATA_FINETUNING_BASE_DIR = os.path.join(PROJECT_ROOT, "data_finetuning")
CORPUS_DIR = os.path.join(DATA_FINETUNING_BASE_DIR, f"corpus_{MODE}")
ASSETS_DIR = os.path.join(CORPUS_DIR, "assets")

PROCESSOR_PATH = os.path.join(ASSETS_DIR, "processor")
TRAIN_DATASET_PATH = os.path.join(ASSETS_DIR, "hf_datasets", "train")
VAL_DATASET_PATH = os.path.join(ASSETS_DIR, "hf_datasets", "validation")

BASE_MODEL_PRETRAINED_NAME = "facebook/wav2vec2-large-xlsr-53"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "finetuning", "finetuning_results", f"xlsr_fon_yoruba_{MODE}")
WANDB_PROJECT_NAME = "master2-asr-fon-yoruba-finetuning"
WANDB_RUN_NAME = f"run_{MODE}_epochs_{TOTAL_TRAIN_EPOCHS}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*20}\n--- LANCEMENT DU FINE-TUNING EN MODE : {MODE.upper()} ---\n{'='*20}")
print(f"Objectif : Atteindre {TOTAL_TRAIN_EPOCHS} époques au total.")
print(f"Utilisation du device : {DEVICE}")

# --- 1. Charger le Processeur et les Datasets ---
print(f"\n[Étape 1/6] Chargement des assets depuis : {ASSETS_DIR}")
try:
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)
    train_dataset = load_from_disk(TRAIN_DATASET_PATH)
    eval_dataset = load_from_disk(VAL_DATASET_PATH)
    print("  Assets chargés avec succès.")
except FileNotFoundError:
    print(f"ERREUR: Assets non trouvés pour le mode '{MODE}'. Avez-vous exécuté prepare_data.py ?"); sys.exit(1)

# --- 2. Définir le Data Collator et les Métriques ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions; pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = pred.label_ids; label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 3. Charger le Modèle (Reprise ou Nouveau) ---
print("\n[Étape 2/6] Configuration du modèle...")
# Vérifier si on doit reprendre un entraînement
resume_from_checkpoint = False
if os.path.isdir(OUTPUT_DIR) and len([d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]) > 0:
    print(f"  Des checkpoints ont été trouvés dans '{OUTPUT_DIR}'. L'entraînement reprendra.")
    resume_from_checkpoint = True
    model_path = OUTPUT_DIR # Le Trainer trouvera le dernier checkpoint
else:
    print(f"  Aucun checkpoint trouvé. Chargement du modèle de base : {BASE_MODEL_PRETRAINED_NAME}")
    model_path = BASE_MODEL_PRETRAINED_NAME

model = Wav2Vec2ForCTC.from_pretrained(
    model_path,
    attention_dropout=0.1, hidden_dropout=0.1, feat_proj_dropout=0.0,
    mask_time_prob=0.05, layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

if hasattr(model, "freeze_feature_extractor"): model.freeze_feature_extractor()
model.gradient_checkpointing_enable()
model.to(DEVICE)
print("  Modèle configuré.")

# --- 4. Définir les Arguments d'Entraînement ---
print("\n[Étape 3/6] Définition des arguments d'entraînement...")
training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  run_name=WANDB_RUN_NAME,
  group_by_length=True,
  per_device_train_batch_size=8,
  gradient_accumulation_steps=2,
  eval_strategy="steps",
  num_train_epochs=TOTAL_TRAIN_EPOCHS, # Objectif final d'époques
  fp16=torch.cuda.is_available(),
  save_steps=500,
  eval_steps=500,
  logging_steps=100,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=3,
  load_best_model_at_end=True,
  metric_for_best_model="wer",
  greater_is_better=False,
  report_to=["wandb"],
  push_to_hub=False,
)

# --- 5. Initialiser le Trainer ---
print("\n[Étape 4/6] Initialisation du Trainer...")
if "wandb" in training_args.report_to:
    try: wandb.init(project=WANDB_PROJECT_NAME, name=training_args.run_name, config=training_args.to_dict(), resume="auto")
    except Exception as e: print(f"Erreur WandB: {e}. Logging désactivé."); training_args.report_to = []

trainer = Trainer(
    model=model, data_collator=data_collator, args=training_args,
    compute_metrics=compute_metrics, train_dataset=train_dataset,
    eval_dataset=eval_dataset, tokenizer=processor.feature_extractor
)
print("  Trainer initialisé.")

# --- 6. Lancer l'Entraînement ---
print("\n[Étape 5/6] Lancement du Fine-tuning...")
try:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("\n[Étape 6/6] Sauvegarde du modèle final et des métriques...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    # ... (le reste du code pour sauvegarder les métriques)
except Exception as e_train:
    print(f"Une ERREUR est survenue pendant l'entraînement: {e_train}"); import traceback; traceback.print_exc()
finally:
    if "wandb" in training_args.report_to and wandb.run is not None: wandb.finish()

print("\n--- Fine-tuning Terminé ---")