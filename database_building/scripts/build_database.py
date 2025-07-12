import pandas as pd

# Define column structures
hate_samples_columns = [
    "Sample_ID", "Group_ID", "Language", "Hate_Type",
    "Text", "Source_Dataset"
]

llm_responses_columns = [
    "Response_ID", "Sample_ID", "Language",
    "Model_Name", "Model_Response"
]

annotations_columns = [
    "Annotation_ID", "Response_ID", "GPT_Label", "Claude_Label",
    "Deepseek_Label", "Llama_Label", "Qwen_Label", "Human_Label", "Final_Label"
]

# Create empty DataFrames with specified columns
hate_samples_df = pd.DataFrame(columns=hate_samples_columns)
llm_responses_df = pd.DataFrame(columns=llm_responses_columns)
annotations_df = pd.DataFrame(columns=annotations_columns)

# Save to CSV files
hate_samples_df.to_csv("hate_samples.csv", index=False)
# llm_responses_df.to_csv("llm_responses.csv", index=False)
# annotations_df.to_csv("annotations.csv", index=False)
