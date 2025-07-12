import pandas as pd
from src.models import LanguageModel
import os

class DataHandler:
    """
    Handles data operations for gathering and annotating hate speech data.
    """
    # --------------------
    # Data Loading Methods
    # --------------------
    def __init__(self, data_path='database_building/data'):
        """
        Initializes the DataHandler with paths to data files.

        Args:
            data_path (str): The path to the data directory.
        """
        self.data_path = data_path
        self.hate_samples_path = os.path.join(data_path, 'hate_samples.csv')
        self.llm_responses_path = os.path.join(data_path, 'llm_responses.csv')
        self.annotations_path = os.path.join(data_path, 'annotations.csv')

    def get_hate_samples(self):
        """
        Reads and returns the hate speech samples from the CSV file.
        """
        return pd.read_csv(self.hate_samples_path)

    def get_llm_responses(self):
        """
        Reads and returns the LLM responses from the CSV file.
        If the file doesn't exist, it returns an empty DataFrame.
        """
        if os.path.exists(self.llm_responses_path) and os.path.getsize(self.llm_responses_path) > 0:
            return pd.read_csv(self.llm_responses_path)
        return pd.DataFrame(columns=['Response_ID', 'Sample_ID', 'Language', 'Model_Name', 'Model_Response'])

    def get_annotations(self):
        """
        Reads and returns the annotations from the CSV file.
        If the file doesn't exist, it returns an empty DataFrame.
        """
        if os.path.exists(self.annotations_path) and os.path.getsize(self.annotations_path) > 0:
            return pd.read_csv(self.annotations_path)
        return pd.DataFrame(columns=['Annotation_ID', 'Response_ID', 'GPT_Label', 'Claude_Label', 'DeepSeek_Label', 'Llama_Label', 'Qwen_Label', 'Human_Label', 'Final_Label'])

    # ---------------------
    # Response Generation
    # ---------------------
    def gather_llm_responses(self, model: LanguageModel, model_name: str, limit: int = None, languages: list = None):
        """
        Gathers responses from a language model for hate speech samples.
        It processes samples language by language to optimize prompt loading.

        Args:
            model (LanguageModel): The language model to use for generating responses.
            model_name (str): The name of the model.
            limit (int, optional): The maximum number of samples to process. Defaults to None (all samples).
            languages (list, optional): A list of languages to process. Defaults to None (all languages).
        """
        # Load hate speech samples and apply limit if provided.
        hate_samples_df = self.get_hate_samples()
        if limit is not None:
            hate_samples_df = hate_samples_df.head(limit)

        # Load existing LLM responses to avoid duplicates.
        llm_responses_df = self.get_llm_responses()
        
        # Create a set of existing responses for efficient lookup.
        existing_responses = set()
        if not llm_responses_df.empty:
            existing_responses = set(zip(llm_responses_df['Sample_ID'], llm_responses_df['Model_Name']))

        # Initialize a list to store new responses.
        new_responses = []
        
        # Determine the starting index for new response IDs.
        start_index = 0
        if not llm_responses_df.empty:
            start_index = llm_responses_df['Response_ID'].max() + 1

        # Get unique languages to process from the samples.
        if languages is None:
            languages = hate_samples_df['Language'].unique()

        # Define batch size for writing to CSV.
        batch_size = 200

        # Process samples for each language separately.
        for lang in languages:
            print(f"Processing language: {lang}")
            model.language = lang.lower()
            try:
                model._load_prompts()
            except ValueError as e:
                print(e)
                print(f"Skipping language {lang} due to unsupported language.")
                continue

            # Filter samples for the current language.
            lang_samples_df = hate_samples_df[hate_samples_df['Language'] == lang]

            # Iterate over each sample for the current language.
            for index, row in lang_samples_df.iterrows():
                sample_id = row['Sample_ID']
                hate_text = row['Text']
                
                # Skip if a response for this sample and model already exists.
                if (sample_id, model_name) in existing_responses:
                    print(f"Skipping Sample_ID {sample_id} for model {model_name} as it already exists.")
                    continue

                # Generate a new response.
                print(f"Generating response for Sample_ID: {sample_id}")
                response_text = model.generate_response(hate_text)

                # Append the new response to the list.
                new_responses.append({
                    'Response_ID': start_index + len(new_responses),
                    'Sample_ID': sample_id,
                    'Language': lang,
                    'Model_Name': model_name,
                    'Model_Response': response_text
                })

                # Write to CSV in batches.
                if len(new_responses) >= batch_size:
                    new_responses_df = pd.DataFrame(new_responses)
                    llm_responses_df = pd.concat([llm_responses_df, new_responses_df], ignore_index=True)
                    llm_responses_df.to_csv(self.llm_responses_path, index=False)
                    print(f"Added {len(new_responses)} new responses to {self.llm_responses_path}")
                    new_responses = [] # Clear the list after writing

        # Save any remaining new responses to the CSV file.
        if new_responses:
            new_responses_df = pd.DataFrame(new_responses)
            llm_responses_df = pd.concat([llm_responses_df, new_responses_df], ignore_index=True)
            llm_responses_df.to_csv(self.llm_responses_path, index=False)
            print(f"Added {len(new_responses)} new responses to {self.llm_responses_path}")

    # ---------------------
    # Response Annotation
    # ---------------------
    def annotate_responses(self, model: LanguageModel, model_name: str, limit: int = None, languages: list = None):
        """
        Annotates responses from a language model.
        It processes responses language by language to optimize prompt loading.

        Args:
            model (LanguageModel): The language model to use for classification.
            model_name (str): The name of the model.
            limit (int, optional): The maximum number of responses to annotate. Defaults to None (all responses).
            languages (list, optional): A list of languages to process. Defaults to None (all languages).
        """
        # Load all necessary dataframes.
        llm_responses_df = self.get_llm_responses()
        hate_samples_df = self.get_hate_samples()
        annotations_df = self.get_annotations()

        # Filter responses for the specified model.
        model_responses_df = llm_responses_df[llm_responses_df['Model_Name'] == model_name].copy()
        
        # Exit if no responses are found for the model.
        if model_responses_df.empty:
            print(f"No responses found for model {model_name} in {self.llm_responses_path}")
            return

        # Merge responses with hate samples to get the full context.
        data_to_annotate = pd.merge(model_responses_df, hate_samples_df, on='Sample_ID')
        
        # Apply limit if provided.
        if limit is not None:
            data_to_annotate = data_to_annotate.head(limit)

        # Determine the column name for the model's labels.
        label_column = f"{model_name.split('-')[0].capitalize()}_Label"
        
        # Initialize a list to store new annotations.
        new_annotations = []
        
        # Get unique languages to process from the data.
        if languages is None:
            languages = data_to_annotate['Language_x'].unique()

        # Define batch size for writing to CSV.
        batch_size = 200

        # Process responses for each language separately.
        for lang in languages:
            print(f"Processing language: {lang}")
            model.language = lang.lower()
            try:
                model._load_prompts()
            except ValueError as e:
                print(e)
                print(f"Skipping language {lang} due to unsupported language.")
                continue

            # Filter data for the current language.
            lang_data_to_annotate = data_to_annotate[data_to_annotate['Language_x'] == lang]

            # Iterate over each response for the current language.
            for index, row in lang_data_to_annotate.iterrows():
                response_id = row['Response_ID']
                
                # Skip if the response is already annotated for this model.
                if not annotations_df[annotations_df['Response_ID'] == response_id].empty:
                    existing_annotation = annotations_df[annotations_df['Response_ID'] == response_id]
                    if label_column in existing_annotation and not pd.isna(existing_annotation[label_column].iloc[0]):
                        print(f"Skipping Response_ID {response_id} for model {model_name} as it is already annotated.")
                        continue

                # Classify the response.
                print(f"Classifying response for Response_ID: {response_id}")
                hate_text = row['Text']
                response_text = row['Model_Response']
                
                classification = model.classify_response(hate_text, response_text)

                # Update existing annotation or create a new one.
                if response_id in annotations_df['Response_ID'].values:
                    annotations_df.loc[annotations_df['Response_ID'] == response_id, label_column] = classification
                else:
                    new_annotation = {'Response_ID': response_id, label_column: classification}
                    start_annotation_id = 0
                    if not annotations_df.empty and not annotations_df['Annotation_ID'].isna().all():
                        start_annotation_id = annotations_df['Annotation_ID'].max() + 1
                    
                    new_annotation['Annotation_ID'] = start_annotation_id + len(new_annotations)
                    new_annotations.append(new_annotation)

                # Write to CSV in batches.
                if len(new_annotations) >= batch_size:
                    new_annotations_df = pd.DataFrame(new_annotations)
                    annotations_df = pd.concat([annotations_df, new_annotations_df], ignore_index=True)
                    annotations_df.to_csv(self.annotations_path, index=False)
                    print(f"Added {len(new_annotations)} new annotations to {self.annotations_path}")
                    new_annotations = [] # Clear the list after writing

        # Add new annotations to the main dataframe.
        if new_annotations:
            new_annotations_df = pd.DataFrame(new_annotations)
            annotations_df = pd.concat([annotations_df, new_annotations_df], ignore_index=True)
        
        # Ensure all label columns exist.
        for col in ['GPT_Label', 'Claude_Label', 'DeepSeek_Label', 'Llama_Label', 'Qwen_Label', 'Human_Label', 'Final_Label']:
            if col not in annotations_df.columns:
                annotations_df[col] = ''
        
        # Save the updated annotations to the CSV file.
        annotations_df.to_csv(self.annotations_path, index=False)
        print(f"Updated annotations in {self.annotations_path}")