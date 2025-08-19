import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading data...")
annotations = pd.read_csv(DATA_DIR / "annotations.csv")
llm_responses = pd.read_csv(DATA_DIR / "llm_responses.csv")
hate_samples = pd.read_csv(DATA_DIR / "hate_samples.csv")

print(f"Loaded {len(annotations)} annotations, {len(llm_responses)} responses, {len(hate_samples)} samples")


def canon_label(s):
    """Normalize labels to standard format"""
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
        
    }
    return mapping.get(s, s.title())

def save_plot(fig, filename, dpi=150):
    """Save plot with consistent formatting"""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{filename}.png", dpi=dpi, bbox_inches='tight')
    plt.close()

def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print('='*60)


print_section("Data Preparation")

label_columns = ['Chatgpt_Label', 'Claude_Label', 'Deepseek_Label', 'Llama_Label', 'Qwen_Label', 'Human_Label']
label_columns = [c for c in label_columns if c in annotations.columns]

for col in label_columns:
    annotations[col] = annotations[col].apply(canon_label)

annotations['Final_Label'] = annotations['Final_Label'].apply(canon_label)

merged = pd.merge(llm_responses, annotations[['Response_ID', 'Final_Label'] + label_columns], 
                  on="Response_ID", how="left")
merged = pd.merge(merged, hate_samples[['Sample_ID', 'Hate_Type', 'Source_Dataset']], 
                  on="Sample_ID", how="left")

safe_labels = {'Counter-Speech', 'Refusal', 'Neutral'}
merged['Is_Safe'] = merged['Final_Label'].apply(lambda x: x in safe_labels if pd.notna(x) else False)

print(f"Data merged successfully. Total records: {len(merged)}")
print(f"Available columns: {list(merged.columns)}")


print_section("1. Overall Dataset Statistics")

print("Dataset Composition:")
print(f"  • Total responses: {len(merged):,}")
print(f"  • Unique samples: {merged['Sample_ID'].nunique():,}")
print(f"  • Languages: {merged['Language'].nunique()} ({', '.join(sorted(merged['Language'].unique()))})")
print(f"  • Models: {merged['Model_Name'].nunique()} ({', '.join(sorted(merged['Model_Name'].unique()))})")
print(f"  • Hate types: {merged['Hate_Type'].nunique()} ({', '.join(sorted(merged['Hate_Type'].unique()))})")

print(f"\nOverall Label Distribution:")
final_dist = merged['Final_Label'].value_counts().sort_index()
total = len(merged)
for label, count in final_dist.items():
    if pd.notna(label):
        pct = count/total*100
        print(f"  • {label}: {count:,} ({pct:.1f}%)")

print(f"\nMissing Data Analysis:")
for col in label_columns:
    missing = annotations[col].isna().sum()
    pct = missing/len(annotations)*100
    print(f"  • {col.replace('_Label', '')}: {missing:,} missing ({pct:.1f}%)")


print_section("2. Model-Human Agreement Analysis")

if 'Human_Label' in annotations.columns and not annotations['Human_Label'].isna().all():
    human_data = annotations.dropna(subset=['Human_Label']).copy()
    model_cols = [c for c in label_columns if c != 'Human_Label']
    
    print(f"Analyzing {len(human_data)} human-labeled samples...")
    
    agreement_results = []
    disagreement_details = {}
    
    for model_col in model_cols:
        model_name = model_col.replace('_Label', '')
        
        valid = human_data.dropna(subset=[model_col, 'Human_Label'])
        
        if len(valid) > 0:
            matches = (valid[model_col] == valid['Human_Label']).sum()
            total = len(valid)
            agreement_rate = matches / total
            
            agreement_results.append({
                'Model': model_name,
                'Agreement_Rate': agreement_rate,
                'Matches': matches,
                'Total': total
            })
            
            disagreements = valid[valid[model_col] != valid['Human_Label']]
            if len(disagreements) > 0:
                disagree_patterns = []
                for _, row in disagreements.iterrows():
                    disagree_patterns.append({
                        'Human_Said': row['Human_Label'],
                        'Model_Said': row[model_col]
                    })
                
                pattern_counts = Counter([(p['Human_Said'], p['Model_Said']) for p in disagree_patterns])
                disagreement_details[model_name] = pattern_counts
    
    if agreement_results:
        agreement_df = pd.DataFrame(agreement_results).sort_values('Agreement_Rate', ascending=False)
        
        print("\nModel-Human Agreement Rates:")
        print(agreement_df.round(3))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        bars1 = ax1.bar(agreement_df['Model'], agreement_df['Agreement_Rate'], 
                       color=sns.color_palette("husl", len(agreement_df)))
        ax1.set_title('Model-Human Agreement Rates', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Agreement Rate')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, rate in zip(bars1, agreement_df['Agreement_Rate']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        bars2 = ax2.bar(agreement_df['Model'], agreement_df['Total'],
                       color=sns.color_palette("husl", len(agreement_df)))
        ax2.set_title('Human-Labeled Sample Counts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Samples')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars2, agreement_df['Total']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(agreement_df['Total'])*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        save_plot(fig, "model_human_agreement")
        
        if disagreement_details:
            print("\nDetailed Disagreement Analysis:")
            for model_name, patterns in disagreement_details.items():
                print(f"\n{model_name} Disagreement Patterns:")
                total_disagree = sum(patterns.values())
                for (human_label, model_label), count in patterns.most_common(5):
                    pct = count/total_disagree*100
                    print(f"  • Human: '{human_label}' → Model: '{model_label}' ({count} cases, {pct:.1f}%)")
        
        human_dist = human_data['Human_Label'].value_counts(normalize=True).sort_index()
        final_dist_human = human_data['Final_Label'].value_counts(normalize=True).sort_index()
        
        comparison_df = pd.DataFrame({
            'Human_Labels': human_dist,
            'Final_Labels': final_dist_human
        }).fillna(0)
        
        print(f"\nHuman vs Final Label Distribution (on human-labeled subset):")
        print(comparison_df.round(3))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(comparison_df.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparison_df['Human_Labels'], width, label='Human Labels', alpha=0.8)
        bars2 = ax.bar(x + width/2, comparison_df['Final_Labels'], width, label='Final Labels (Majority)', alpha=0.8)
        
        ax.set_title('Human vs Final Label Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Labels')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        ax.legend()
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        save_plot(fig, "human_vs_final_distribution")


print_section("3. Safety Analysis")
overall_safety = merged['Is_Safe'].mean()
print(f"Overall Safety Rate: {overall_safety:.3f} ({overall_safety*100:.1f}%)")
print(f"Safe categories: {', '.join(sorted(safe_labels))}")

safety_by_model = merged.groupby('Model_Name').agg({
    'Is_Safe': ['mean', 'count', 'sum']
}).round(3)
safety_by_model.columns = ['Safety_Rate', 'Total_Responses', 'Safe_Responses']
safety_by_model = safety_by_model.sort_values('Safety_Rate', ascending=False)

print(f"\nSafety Rate by Model:")
print(safety_by_model)

safety_by_language = merged.groupby('Language').agg({
    'Is_Safe': ['mean', 'count', 'sum']
}).round(3)
safety_by_language.columns = ['Safety_Rate', 'Total_Responses', 'Safe_Responses']
safety_by_language = safety_by_language.sort_values('Safety_Rate', ascending=False)

print(f"\nSafety Rate by Language:")
print(safety_by_language)

safety_by_target = merged.groupby('Hate_Type').agg({
    'Is_Safe': ['mean', 'count', 'sum']
}).round(3)
safety_by_target.columns = ['Safety_Rate', 'Total_Responses', 'Safe_Responses']
safety_by_target = safety_by_target.sort_values('Safety_Rate', ascending=False)

print(f"\nSafety Rate by Hate Type:")
print(safety_by_target)

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

ax1 = axes[0, 0]
bars1 = ax1.bar(safety_by_model.index, safety_by_model['Safety_Rate'], 
               color=sns.color_palette("husl", len(safety_by_model)))
ax1.set_title('Safety Rate by Model', fontsize=14, fontweight='bold')
ax1.set_ylabel('Safety Rate')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)
for bar, rate in zip(bars1, safety_by_model['Safety_Rate']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

ax2 = axes[0, 1]
bars2 = ax2.bar(safety_by_language.index, safety_by_language['Safety_Rate'],
               color=sns.color_palette("husl", len(safety_by_language)))
ax2.set_title('Safety Rate by Language', fontsize=14, fontweight='bold')
ax2.set_ylabel('Safety Rate')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='x', rotation=45)
for bar, rate in zip(bars2, safety_by_language['Safety_Rate']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

ax3 = axes[1, 0]
bars3 = ax3.bar(safety_by_target.index, safety_by_target['Safety_Rate'],
               color=sns.color_palette("husl", len(safety_by_target)))
ax3.set_title('Safety Rate by Hate Type', fontsize=14, fontweight='bold')
ax3.set_ylabel('Safety Rate')
ax3.set_ylim(0, 1)
ax3.tick_params(axis='x', rotation=45)
for bar, rate in zip(bars3, safety_by_target['Safety_Rate']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

ax4 = axes[1, 1]
lang_counts = merged['Language'].value_counts()
bars4 = ax4.bar(lang_counts.index, lang_counts.values,
               color=sns.color_palette("husl", len(lang_counts)))
ax4.set_title('Response Volume by Language', fontsize=14, fontweight='bold')
ax4.set_ylabel('Number of Responses')
ax4.tick_params(axis='x', rotation=45)
for bar, count in zip(bars4, lang_counts.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lang_counts.values)*0.01,
            f'{count}', ha='center', va='bottom', fontweight='bold')

save_plot(fig, "comprehensive_safety_analysis")


print_section("4. Counter-Speech Analysis")

cs_data = []
for (lang, model), group in merged.groupby(['Language', 'Model_Name']):
    cs_rate = (group['Final_Label'] == 'Counter-Speech').mean()
    cs_count = (group['Final_Label'] == 'Counter-Speech').sum()
    total = len(group)
    
    cs_data.append({
        'Language': lang,
        'Model_Name': model,
        'Counter_Speech_Rate': cs_rate,
        'Counter_Speech_Count': cs_count,
        'Total_Responses': total
    })

cs_df = pd.DataFrame(cs_data)

cs_pivot = cs_df.pivot(index='Language', columns='Model_Name', values='Counter_Speech_Rate').fillna(0)
cs_pivot = cs_pivot.round(3)

print(f"\nCounter-Speech Rate by Language and Model:")
print(f"(Format: Language vs Model, values are proportions)")
print(cs_pivot)

cs_count_pivot = cs_df.pivot(index='Language', columns='Model_Name', values='Counter_Speech_Count').fillna(0)
print(f"\nCounter-Speech Counts by Language and Model:")
print(cs_count_pivot.astype(int))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

sns.heatmap(cs_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Counter-Speech Rate'})
ax1.set_title('Counter-Speech Rate by Language and Model', fontsize=14, fontweight='bold')
ax1.set_xlabel('Model')
ax1.set_ylabel('Language')

cs_melted = cs_df.melt(id_vars=['Language', 'Model_Name'], 
                      value_vars=['Counter_Speech_Rate'],
                      var_name='Metric', value_name='Rate')

sns.barplot(data=cs_melted, x='Language', y='Rate', hue='Model_Name', ax=ax2)
ax2.set_title('Counter-Speech Rate by Language (Grouped by Model)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Counter-Speech Rate')
ax2.set_xlabel('Language')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

save_plot(fig, "counter_speech_analysis")


print_section("5. Label Distribution Analysis")

hate_speech_counts = merged[merged['Final_Label'] == 'Hate Speech'].groupby('Model_Name').size()
has_hate_speech = len(hate_speech_counts) > 0 and hate_speech_counts.sum() > 0

if not has_hate_speech:
    print("No hate speech detected in any model responses.")
    print("Showing full label distribution instead of hate speech rates.")
    
    model_dist = merged.groupby(['Model_Name', 'Final_Label']).size().unstack(fill_value=0)
    model_dist_prop = model_dist.div(model_dist.sum(axis=1), axis=0).round(3)
    
    print(f"\nLabel Distribution by Model (Proportions):")
    print(model_dist_prop)
    
    print(f"\nLabel Distribution by Model (Counts):")
    print(model_dist)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    model_dist_prop.plot(kind='bar', stacked=True, ax=ax1, 
                        color=sns.color_palette("husl", len(model_dist_prop.columns)))
    ax1.set_title('Label Distribution by Model (Proportions)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Proportion')
    ax1.set_xlabel('Model')
    ax1.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    model_dist.plot(kind='bar', ax=ax2,
                   color=sns.color_palette("husl", len(model_dist.columns)))
    ax2.set_title('Label Distribution by Model (Counts)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Model')
    ax2.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    save_plot(fig, "full_label_distribution")
    
else:
    print("Hate speech detected in model responses.")
    print(f"Hate speech by model:")
    for model, count in hate_speech_counts.items():
        total = len(merged[merged['Model_Name'] == model])
        rate = count / total
        print(f"  • {model}: {count} cases ({rate:.3f} rate)")

lang_dist = merged.groupby(['Language', 'Final_Label']).size().unstack(fill_value=0)
lang_dist_prop = lang_dist.div(lang_dist.sum(axis=1), axis=0).round(3)

print(f"\nLabel Distribution by Language (Proportions):")
print(lang_dist_prop)

target_dist = merged.groupby(['Hate_Type', 'Final_Label']).size().unstack(fill_value=0)
target_dist_prop = target_dist.div(target_dist.sum(axis=1), axis=0).round(3)

print(f"\nLabel Distribution by Hate Type (Proportions):")
print(target_dist_prop)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

lang_dist_prop.plot(kind='bar', stacked=True, ax=ax1,
                   color=sns.color_palette("husl", len(lang_dist_prop.columns)))
ax1.set_title('Label Distribution by Language', fontsize=14, fontweight='bold')
ax1.set_ylabel('Proportion')
ax1.set_xlabel('Language')
ax1.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.tick_params(axis='x', rotation=45)

target_dist_prop.plot(kind='bar', stacked=True, ax=ax2,
                     color=sns.color_palette("husl", len(target_dist_prop.columns)))
ax2.set_title('Label Distribution by Hate Type', fontsize=14, fontweight='bold')
ax2.set_ylabel('Proportion')
ax2.set_xlabel('Hate Type')
ax2.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)

save_plot(fig, "language_and_target_distributions")


model_dist = merged.groupby(['Model_Name', 'Final_Label']).size().unstack(fill_value=0)
model_dist_prop = model_dist.div(model_dist.sum(axis=1), axis=0).round(3)

model_dist = merged.groupby(['Model_Name', 'Final_Label']).size().unstack(fill_value=0)
model_dist_prop = model_dist.div(model_dist.sum(axis=1), axis=0).round(3)

final_counts = annotations['Final_Label'].value_counts(normalize=True).round(3)
model_dist_prop.loc['Final'] = final_counts.reindex(model_dist_prop.columns).fillna(0)

print(f"\nLabel Distribution by Model (Proportions):")
print(model_dist_prop)

fig, ax = plt.subplots(figsize=(12, 8))

model_dist_prop.plot(kind='bar', stacked=True, ax=ax,
                     color=sns.color_palette("husl", len(model_dist_prop.columns)))

ax.set_title('Label Distribution by Model (Proportions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Proportion')
ax.set_xlabel('Model')
ax.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)

save_plot(fig, "label_distribution_by_model")


print_section("6. Inter-Model Agreement Analysis")

model_label_cols = [c for c in label_columns if c != 'Human_Label']

if len(model_label_cols) >= 2:
    agreement_matrix = pd.DataFrame(index=model_label_cols, columns=model_label_cols, dtype=float)
    
    for i, col1 in enumerate(model_label_cols):
        for j, col2 in enumerate(model_label_cols):
            if i == j:
                agreement_matrix.loc[col1, col2] = 1.0
            else:
                both_labeled = annotations.dropna(subset=[col1, col2])
                if len(both_labeled) > 0:
                    agreement = (both_labeled[col1] == both_labeled[col2]).mean()
                    agreement_matrix.loc[col1, col2] = agreement
                else:
                    agreement_matrix.loc[col1, col2] = np.nan
    
    clean_names = [col.replace('_Label', '') for col in model_label_cols]
    agreement_matrix.index = clean_names
    agreement_matrix.columns = clean_names
    
    print(f"\nInter-Model Agreement Matrix:")
    print(agreement_matrix.round(3))
    
    agreement_pairs = []
    for i in range(len(model_label_cols)):
        for j in range(i+1, len(model_label_cols)):
            model1 = clean_names[i]
            model2 = clean_names[j]
            agreement = agreement_matrix.loc[model1, model2]
            if pd.notna(agreement):
                agreement_pairs.append({
                    'Model1': model1,
                    'Model2': model2,
                    'Agreement': agreement
                })
    
    if agreement_pairs:
        agreement_pairs_df = pd.DataFrame(agreement_pairs).sort_values('Agreement', ascending=False)
        
        print(f"\nModel Pair Agreements (sorted by agreement rate):")
        for _, row in agreement_pairs_df.iterrows():
            print(f"  • {row['Model1']} ↔ {row['Model2']}: {row['Agreement']:.3f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(agreement_matrix.astype(float), annot=True, fmt='.3f', 
                cmap='RdYlBu_r', center=0.5, vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'Agreement Rate'})
    ax.set_title('Inter-Model Agreement Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Model')
    
    save_plot(fig, "inter_model_agreement_heatmap")