import pandas as pd
import csv
import os
import json

def merge_conan_to_hate_samples():
    """
    Merge all CONAN datasets into the hate samples file.
    
    CONAN schemas: INDEX, HATE_SPEECH, COUNTER_NARRATIVE, TARGET, VERSION
    Hate samples schema: Sample_ID, Group_ID, Language, Hate_Type, Text, Source_Dataset
    """
    
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up to database_building/
    project_root = os.path.dirname(project_root)  # Go up to project/
    
    # Define paths
    data_dir = os.path.join(project_root, 'database_building', 'data')
    conan_dir = os.path.join(project_root, 'dataset', 'conan')
    hate_samples_path = os.path.join(data_dir, 'hate_samples.csv')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # First, let's discover what files actually exist
    print(f"Looking for files in: {conan_dir}")
    if os.path.exists(conan_dir):
        actual_files = os.listdir(conan_dir)
        print(f"Found files: {actual_files}")
    else:
        print(f"Directory {conan_dir} does not exist!")
        return
    
    # Define the exact CONAN datasets based on discovered files
    # Using only CSV files to avoid duplicates
    conan_datasets = {
        'Albanian_CONAN.csv': 'Albanian',
        'Arabic_CONAN.csv': 'Arabic', 
        'Chinese_CONAN.csv': 'Chinese',
        'Multitarget-CONAN.csv': 'English'
    }
    
    # Filter to only include files that actually exist
    existing_datasets = {filename: language for filename, language in conan_datasets.items() 
                        if filename in actual_files}
    
    print(f"Will process these datasets: {existing_datasets}")
    
    # Read existing hate samples to get the next available Sample_ID
    try:
        hate_samples = pd.read_csv(hate_samples_path)
        next_sample_id = hate_samples['Sample_ID'].max() + 1 if not hate_samples.empty else 1
    except FileNotFoundError:
        # If file doesn't exist, start from 1
        next_sample_id = 1
        # Create the file with headers
        with open(hate_samples_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_ID', 'Group_ID', 'Language', 'Hate_Type', 'Text', 'Source_Dataset'])
    
    # First pass: collect all Group_IDs from Albanian, Arabic, and Chinese datasets
    allowed_group_ids = set()
    
    for filename, language in existing_datasets.items():
        if filename != 'Multitarget-CONAN.csv':  # Skip multi-target for now
            file_path = os.path.join(conan_dir, filename)
            
            try:
                print(f"Collecting Group_IDs from {filename}...")
                
                if filename == 'Albanian_CONAN.csv':
                    # Handle Albanian file parsing
                    try:
                        temp_df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
                    except:
                        try:
                            temp_df = pd.read_csv(file_path, encoding='utf-8', sep=',', quotechar='"', on_bad_lines='skip')
                        except:
                            import csv
                            rows = []
                            with open(file_path, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                headers = next(reader)
                                for i, row in enumerate(reader):
                                    if len(row) >= 5:
                                        rows.append(row[:5])
                                    elif len(row) > 0:
                                        rows.append(row + [''] * (5 - len(row)))
                            temp_df = pd.DataFrame(rows, columns=headers[:5])
                else:
                    temp_df = pd.read_csv(file_path, encoding='utf-8')
                
                # Clean column names
                temp_df.columns = [col.strip() for col in temp_df.columns]
                
                if 'INDEX' in temp_df.columns:
                    allowed_group_ids.update(temp_df['INDEX'].tolist())
                    print(f"  Added {len(temp_df)} Group_IDs from {filename}")
                
            except Exception as e:
                print(f"Error collecting Group_IDs from {filename}: {str(e)}")
    
    print(f"Total allowed Group_IDs: {len(allowed_group_ids)}")
    
    # Process each CONAN dataset
    for filename, language in existing_datasets.items():
        file_path = os.path.join(conan_dir, filename)
        
        try:
            print(f"Processing {filename}...")
            
            # All files are now CSV files
            if filename == 'Albanian_CONAN.csv':
                # Try different approaches for the problematic Albanian file
                try:
                    conan_df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
                except:
                    try:
                        conan_df = pd.read_csv(file_path, encoding='utf-8', sep=',', quotechar='"', on_bad_lines='skip')
                    except:
                        try:
                            conan_df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
                        except:
                            print(f"Could not parse {filename} with standard methods. Trying manual parsing...")
                            # Manual parsing for problematic file
                            import csv
                            rows = []
                            with open(file_path, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                headers = next(reader)
                                for i, row in enumerate(reader):
                                    if len(row) >= 5:  # Take first 5 columns
                                        rows.append(row[:5])
                                    elif len(row) > 0:  # Skip empty rows
                                        # Pad with empty strings if needed
                                        rows.append(row + [''] * (5 - len(row)))
                            conan_df = pd.DataFrame(rows, columns=headers[:5])
            else:
                # Standard CSV reading for other files
                conan_df = pd.read_csv(file_path, encoding='utf-8')
            
            if conan_df.empty:
                print(f"Warning: {filename} is empty. Skipping...")
                continue
            
            # Check what columns we have
            print(f"Available columns in {filename}: {list(conan_df.columns)}")
            print(f"Number of rows: {len(conan_df)}")
            
            # Ensure we have INDEX column
            if 'INDEX' not in conan_df.columns:
                print(f"Adding INDEX column to {filename}")
                conan_df['INDEX'] = range(len(conan_df))
            
            # Check if we have the essential columns
            required_columns = ['HATE_SPEECH', 'TARGET']
            available_columns = list(conan_df.columns)
            
            # Clean up column names (remove extra spaces)
            conan_df.columns = [col.strip() for col in conan_df.columns]
            
            missing_columns = [col for col in required_columns if col not in conan_df.columns]
            
            if missing_columns:
                print(f"Warning: {filename} missing essential columns: {missing_columns}")
                print(f"Available columns: {list(conan_df.columns)}")
                continue
            
            # For multi-target dataset, filter to only include entries present in other datasets
            if filename == 'Multitarget-CONAN.csv':
                print(f"Filtering multi-target entries to only include Group_IDs present in other datasets...")
                original_count = len(conan_df)
                conan_df = conan_df[conan_df['INDEX'].isin(allowed_group_ids)]
                filtered_count = len(conan_df)
                print(f"  Filtered from {original_count} to {filtered_count} entries")
            
            # Prepare data for hate samples format
            new_entries = []
            
            for _, row in conan_df.iterrows():
                # Handle missing values
                index_val = row.get('INDEX', next_sample_id)
                target_val = row.get('TARGET', 'Unknown')
                hate_speech_val = row.get('HATE_SPEECH', '')
                
                # Skip empty hate speech entries
                if pd.isna(hate_speech_val) or str(hate_speech_val).strip() == '':
                    continue
                
                new_entry = {
                    'Sample_ID': next_sample_id,
                    'Group_ID': index_val,
                    'Language': language,
                    'Hate_Type': target_val,
                    'Text': str(hate_speech_val).strip(),
                    'Source_Dataset': 'CONAN'  # All entries now have same source
                }
                new_entries.append(new_entry)
                next_sample_id += 1
            
            if new_entries:
                # Append to hate samples CSV
                new_df = pd.DataFrame(new_entries)
                new_df.to_csv(hate_samples_path, mode='a', header=False, index=False, encoding='utf-8')
                
                print(f"Successfully added {len(new_entries)} entries from {filename}")
            else:
                print(f"No valid entries found in {filename}")
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found at {file_path}. Skipping...")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("\nMerge complete!")
    
    # Display summary statistics
    try:
        final_hate_samples = pd.read_csv(hate_samples_path)
        print(f"\nFinal hate_samples.csv contains {len(final_hate_samples)} total entries")
        print(f"File saved at: {hate_samples_path}")
        
        # Show breakdown by source dataset
        print("\nBreakdown by source dataset:")
        source_counts = final_hate_samples['Source_Dataset'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} entries")
            
        # Show breakdown by language
        print("\nBreakdown by language:")
        language_counts = final_hate_samples['Language'].value_counts()
        for language, count in language_counts.items():
            print(f"  {language}: {count} entries")
            
    except Exception as e:
        print(f"Error generating summary: {str(e)}")

def preview_data():
    """
    Preview the first few entries from each dataset to verify the merge.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'database_building', 'data')
    hate_samples_path = os.path.join(data_dir, 'hate_samples.csv')
    
    try:
        hate_samples = pd.read_csv(hate_samples_path)
        print("Preview of hate_samples.csv:")
        print(hate_samples.head())
        print(f"\nTotal entries: {len(hate_samples)}")
        
        # Show sample from each source
        print("\nSample from each source:")
        for source in hate_samples['Source_Dataset'].unique():
            sample = hate_samples[hate_samples['Source_Dataset'] == source].iloc[0]
            print(f"\n{source}:")
            print(f"  Sample_ID: {sample['Sample_ID']}")
            print(f"  Language: {sample['Language']}")
            print(f"  Hate_Type: {sample['Hate_Type']}")
            print(f"  Text: {sample['Text'][:100]}...")  # First 100 chars
            
    except FileNotFoundError:
        print("hate_samples.csv not found. Run merge_conan_to_hate_samples() first.")

if __name__ == "__main__":
    # Run the merge
    merge_conan_to_hate_samples()
    
    # Preview the results
    print("\n" + "="*50)
    preview_data()