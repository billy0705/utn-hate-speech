# Hate Speech Database - What I Built so far

I merged the translations (Albanian, Arabic, Chinese) and the original: Multi-target into a single database structure with 3 main files:

### Files Created:
- **`hate_samples.csv`** - Main dataset with all hate speech samples
- **`llm_responses.csv`** - Empty, will be populated when we have the llm responses
- **`annotations.csv`** - Empty, will be populated when we have the llm responses

### Database Structure:
```
hate_samples.csv:
Sample_ID | Group_ID | Language | Hate_Type | Text | Source_Dataset

llm_responses.csv:
Response_ID | Sample_ID | Language | Model_Name | Model_Response

annotations.csv:
Annotation_ID | Response_ID | GPT_Label | Claude_Label | DeepSeek_Label | Llama_Label | Qwen_Label | Human_Label | Final_Label
```

## Key Points:

1. **Group_ID is important** - it links the same hate speech content across different languages
2. **Multi-target dataset was filtered** - only kept entries that have corresponding Group_IDs in other languages

## Scripts You Can Use:

- **`populate_hates_samples.py`** - Main merging script (already run)
- **`build_database.py`** - Creates empty CSV files with headers
- **`check_files.py`** - Analyzes which Group_IDs exist in which languages
