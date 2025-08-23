import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns
import matplotlib.cm as cm

from collections import Counter

def add_majority_vote(annotations_df):
    model_cols = ["Chatgpt_Label", "Claude_Label", "Deepseek_Label", "Llama_Label", "Qwen_Label"]

    def majority_vote(row):
        votes = [row[c] for c in model_cols if pd.notnull(row[c])]
        if not votes:
            return None
        counter = Counter(votes)
        # Get the label(s) with max count
        most_common = counter.most_common()
        max_count = most_common[0][1]
        winners = [label for label, count in most_common if count == max_count]
        if len(winners) == 1:
            return winners[0]
        else:
            # Tie case: you can choose policy
            # e.g. prefer Human_Label if available, else pick first alphabetically
            if "Human_Label" in row and pd.notnull(row["Human_Label"]):
                return row["Human_Label"]
            return sorted(winners)[0]

    annotations_df["Final_Label"] = annotations_df.apply(majority_vote, axis=1)
    return annotations_df

matplotlib.rcParams.update(
    {
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 'medium',
        'figure.titlesize': 'medium',
        'text.usetex': True,
        'text.latex.preamble': '\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage{siunitx}[=v2]',
        'pgf.rcfonts': False,
        'pgf.texsystem': 'pdflatex'
    }
)

def create_plots_directory(directory):
    # Create plots directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_dataset_target_distribution(df, output_root="../plots/pdf"):
    # only choose the Language is English
    df = df[df['Language'] == 'English']
    # group by Hate_Type and count occurrences
    hate_type_counts = df['Hate_Type'].value_counts().tolist()
    Type = df['Hate_Type'].value_counts().index.tolist()
    Type = sorted(Type)
    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute:d})"

    N = len(hate_type_counts)
    cmap = cm.get_cmap("Blues")
    colors = [cmap(x) for x in np.linspace(0.3, 1, N)]

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(hate_type_counts, autopct=lambda pct: func(pct, hate_type_counts),
                                    textprops=dict(color="w"), colors=colors)

    ax.legend(wedges, Type,
            title="Hate Target",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    # ax.set_title("Data Distribution")
    plt.tight_layout()
    output_path = os.path.join(output_root, "data_distribution.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    # plt.show()

def plot_language_annotation(annotations_df, llm_responses_df, hate_samples_df, output_root="../plots/"):
    # Merge the dataframes
    annotations_df = add_majority_vote(annotations_df)
    # --- Merge to attach Sample_ID & Language ---
    merged = pd.merge(annotations_df, llm_responses_df[['Response_ID','Sample_ID']], on='Response_ID', how='left')
    merged = pd.merge(merged, hate_samples_df[['Sample_ID','Language']], on='Sample_ID', how='left')

    # --- Deduplicate so each sample counts once per (Language, Final_Label) ---
    dedup = merged[['Language','Final_Label','Sample_ID']].dropna().drop_duplicates()

    # --- Pivot: rows = Language, cols = Final_Label ---
    table_df = (
        dedup
        .pivot_table(index='Language',
                     columns='Final_Label',
                     values='Sample_ID',
                     aggfunc=pd.Series.nunique,
                     fill_value=0)
        .sort_index()
    )

    # --- Plot as matplotlib table ---
    n_rows, n_cols = table_df.shape
    fig_w = max(6, min(12, 1.2 * n_cols + 3))   # heuristic width
    fig_h = max(3, 0.5 * n_rows + 2)            # heuristic height

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index.tolist(),
        colLabels=table_df.columns.tolist(),
        loc='center',
        cellLoc='center',
        rowLoc='center'
    )

    tbl.auto_set_font_size(False)
    base_fs = 12
    if n_rows * n_cols > 80:
        base_fs = 9
    if n_rows * n_cols > 150:
        base_fs = 8
    tbl.set_fontsize(base_fs)
    tbl.scale(1, 1.2)

    ax.set_title("Language × Final_Label (unique Sample_ID counts)", fontsize=14, pad=12)

    # Save
    os.makedirs(output_root, exist_ok=True)
    outpath = os.path.join(output_root, "language_by_final_label.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    # plt.show()

    print(f"Saved table figure to: {outpath}")

# def plot_analysis():

#     # Load the datasets
#     annotations_df = pd.read_csv('dataset/data/annotations.csv')
#     llm_responses_df = pd.read_csv('dataset/data/llm_responses.csv')
#     hate_samples_df = pd.read_csv('dataset/data/hate_samples.csv')

#     # Merge the dataframes
#     merged_df = pd.merge(annotations_df, llm_responses_df, on='Response_ID')
#     merged_df = pd.merge(merged_df, hate_samples_df, on='Sample_ID')

#     # Analysis of labels by model (percentage)
#     model_labels = merged_df.melt(id_vars=['Model_Name'], value_vars=['Chatgpt_Label', 'Claude_Label', 'Deepseek_Label', 'Llama_Label', 'Qwen_Label'],
#                                   var_name='Annotator', value_name='Label')
#     model_percentages = model_labels.groupby('Model_Name')['Label'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=model_percentages, x='Model_Name', y='Percentage', hue='Label', zorder=2)
#     plt.title('Distribution of Labels by Model (Percentage)')
#     plt.xticks(rotation=45)
#     plt.ylabel('Percentage')
#     plt.grid(zorder=0)
#     plt.tight_layout()
#     plt.savefig('plots/labels_by_model_percentage.png')
#     print("Plot 'plots/labels_by_model_percentage.png' saved.")

#     # Analysis of labels by language (percentage)
#     language_percentages = merged_df.groupby('Language_x')['Chatgpt_Label'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=language_percentages, x='Language_x', y='Percentage', hue='Chatgpt_Label', zorder=2)
#     plt.title('Distribution of ChatGPT Labels by Language (Percentage)')
#     plt.xticks(rotation=45)
#     plt.ylabel('Percentage')
#     plt.grid(zorder=0)
#     plt.tight_layout()
#     plt.savefig('plots/labels_by_language_percentage.png')
#     print("Plot 'plots/labels_by_language_percentage.png' saved.")

#     # Analysis of labels by hate target (percentage)
#     hate_target_percentages = merged_df.groupby('Hate_Type')['Chatgpt_Label'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=hate_target_percentages, x='Hate_Type', y='Percentage', hue='Chatgpt_Label', zorder=2)
#     plt.title('Distribution of ChatGPT Labels by Hate Target (Percentage)')
#     plt.xticks(rotation=45)
#     plt.ylabel('Percentage')
#     plt.grid(zorder=0)
#     plt.tight_layout()
#     plt.savefig('plots/labels_by_hate_target_percentage.png')
#     print("Plot 'plots/labels_by_hate_target_percentage.png' saved.")


if __name__ == "__main__":
    df = pd.read_csv("../dataset/data/hate_samples.csv")
    plot_dataset_target_distribution(df, output_root="../plots/")