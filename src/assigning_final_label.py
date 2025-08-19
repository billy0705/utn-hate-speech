import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent  
DATA_DIR = BASE_DIR / "dataset" / "data"


annotations = pd.read_csv(DATA_DIR / "annotations.csv")

print(f"Loaded {len(annotations)} annotations")


def canon_label(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "hate speech": "Hate Speech",
        "hatespeech": "Hate Speech",
        "counter speech": "Counter-Speech",
        "counter speech.": "Counter-Speech",
        "counter-speech": "Counter-Speech",
        "refusal": "Refusal",
        "neutral": "Neutral",
        "stop": "Stop",
    }
    return mapping.get(s, s.title())


label_columns = [
    'Chatgpt_Label', 'Claude_Label', 'Deepseek_Label',
    'Llama_Label', 'Qwen_Label', 'Human_Label'
]

label_columns = [c for c in label_columns if c in annotations.columns]
print(f"Using label columns: {label_columns}")

def majority_label(row):
    """Calculate majority label from available model/human labels"""
    if not label_columns:
        return None
    
    labels = []
    for col in label_columns:
        if pd.notna(row[col]):
            normalized = canon_label(row[col])
            if normalized is not None:
                labels.append(normalized)
    
    if not labels:
        return None
    
    label_counts = Counter(labels)
    most_common = label_counts.most_common(1)
    return most_common[0][0] if most_common else None

print("\nCalculating final labels...")
annotations['Final_Label'] = annotations.apply(majority_label, axis=1)

print(f"\nFinal Label Statistics:")
print(f"- Total annotations: {len(annotations)}")
print(f"- Annotations with final label: {annotations['Final_Label'].notna().sum()}")
print(f"- Annotations without sufficient labels: {annotations['Final_Label'].isna().sum()}")

final_dist = annotations['Final_Label'].value_counts().sort_index()
print(f"\nFinal Label Distribution:")
for label, count in final_dist.items():
    if pd.notna(label):
        print(f"  {label}: {count} ({count/len(annotations)*100:.1f}%)")


annotations.to_csv(DATA_DIR / "annotations.csv", index=False)
print(f"\n Original annotations.csv updated with Final_Label column")
print(f"Updated file: {DATA_DIR / 'annotations.csv'}")
print(f" Final Label populated successfully!")