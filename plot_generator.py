

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_analysis():
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Load the datasets
    annotations_df = pd.read_csv('database_building/data/annotations.csv')
    llm_responses_df = pd.read_csv('database_building/data/llm_responses.csv')
    hate_samples_df = pd.read_csv('database_building/data/hate_samples.csv')

    # Merge the dataframes
    merged_df = pd.merge(annotations_df, llm_responses_df, on='Response_ID')
    merged_df = pd.merge(merged_df, hate_samples_df, on='Sample_ID')

    # Analysis of labels by model (percentage)
    model_labels = merged_df.melt(id_vars=['Model_Name'], value_vars=['Chatgpt_Label', 'Claude_Label', 'Deepseek_Label', 'Llama_Label', 'Qwen_Label'],
                                  var_name='Annotator', value_name='Label')
    model_percentages = model_labels.groupby('Model_Name')['Label'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=model_percentages, x='Model_Name', y='Percentage', hue='Label', zorder=2)
    plt.title('Distribution of Labels by Model (Percentage)')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig('plots/labels_by_model_percentage.png')
    print("Plot 'plots/labels_by_model_percentage.png' saved.")

    # Analysis of labels by language (percentage)
    language_percentages = merged_df.groupby('Language_x')['Chatgpt_Label'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=language_percentages, x='Language_x', y='Percentage', hue='Chatgpt_Label', zorder=2)
    plt.title('Distribution of ChatGPT Labels by Language (Percentage)')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig('plots/labels_by_language_percentage.png')
    print("Plot 'plots/labels_by_language_percentage.png' saved.")

    # Analysis of labels by hate target (percentage)
    hate_target_percentages = merged_df.groupby('Hate_Type')['Chatgpt_Label'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=hate_target_percentages, x='Hate_Type', y='Percentage', hue='Chatgpt_Label', zorder=2)
    plt.title('Distribution of ChatGPT Labels by Hate Target (Percentage)')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig('plots/labels_by_hate_target_percentage.png')
    print("Plot 'plots/labels_by_hate_target_percentage.png' saved.")

if __name__ == '__main__':
    plot_analysis()

