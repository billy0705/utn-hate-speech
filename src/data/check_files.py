import pandas as pd
import os
from collections import defaultdict

def analyze_dataset_correspondence():
    """
    Analyze the correspondence between different language datasets.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'database_building', 'data')
    hate_samples_path = os.path.join(data_dir, 'hate_samples.csv')
    
    try:
        df = pd.read_csv(hate_samples_path)
        print(f"Loaded {len(df)} entries from hate_samples.csv")
        
        language_groups = {}
        for language in df['Language'].unique():
            lang_df = df[df['Language'] == language]
            language_groups[language] = set(lang_df['Group_ID'].unique())
            print(f"{language}: {len(lang_df)} entries, {len(language_groups[language])} unique Group_IDs")
        
        all_languages = list(language_groups.keys())
        if len(all_languages) > 1:
            common_group_ids = set.intersection(*language_groups.values())
            print(f"\nCommon Group_IDs across all languages: {len(common_group_ids)}")
            
            print("\nGroup_IDs unique to each language:")
            for language in all_languages:
                unique_to_lang = language_groups[language] - set.union(*[language_groups[lang] for lang in all_languages if lang != language])
                print(f"  {language}: {len(unique_to_lang)} unique Group_IDs")
                if len(unique_to_lang) > 0 and len(unique_to_lang) <= 10:
                    print(f"    Examples: {list(unique_to_lang)[:5]}")
        
        print("\nGroup_IDs missing from each language:")
        all_group_ids = set.union(*language_groups.values())
        
        for language in all_languages:
            missing_from_lang = all_group_ids - language_groups[language]
            print(f"  {language}: missing {len(missing_from_lang)} Group_IDs")
            if len(missing_from_lang) > 0 and len(missing_from_lang) <= 10:
                print(f"    Examples: {list(missing_from_lang)[:5]}")
        
        print(f"\nCorrespondence Matrix:")
        print(f"{'Language':<15} {'Total':<8} {'Common':<8} {'Missing':<8} {'Unique':<8}")
        print("-" * 50)
        
        for language in all_languages:
            total = len(language_groups[language])
            common = len(language_groups[language] & common_group_ids) if len(all_languages) > 1 else total
            missing = len(all_group_ids - language_groups[language])
            unique = len(language_groups[language] - set.union(*[language_groups[lang] for lang in all_languages if lang != language]))
            print(f"{language:<15} {total:<8} {common:<8} {missing:<8} {unique:<8}")
        
        print(f"\nDetailed Analysis of Non-Common Group_IDs:")
        print("-" * 60)
        
        partial_group_ids = all_group_ids - common_group_ids
        
        if len(partial_group_ids) > 0:
            print(f"Group_IDs that don't appear in all languages: {len(partial_group_ids)}")
            
            for group_id in sorted(list(partial_group_ids))[:10]: 
                languages_with_group = [lang for lang in all_languages if group_id in language_groups[lang]]
                languages_without_group = [lang for lang in all_languages if group_id not in language_groups[lang]]
                
                print(f"\nGroup_ID {group_id}:")
                print(f"  Present in: {', '.join(languages_with_group)}")
                print(f"  Missing from: {', '.join(languages_without_group)}")
                
                sample_entry = df[df['Group_ID'] == group_id].iloc[0]
                print(f"  Sample text: {sample_entry['Text'][:100]}...")
                print(f"  Hate_Type: {sample_entry['Hate_Type']}")
        
        return {
            'language_groups': language_groups,
            'common_group_ids': common_group_ids if len(all_languages) > 1 else set(),
            'total_group_ids': len(all_group_ids),
            'correspondence_complete': len(common_group_ids) == len(all_group_ids) if len(all_languages) > 1 else True
        }
        
    except FileNotFoundError:
        print(f"Error: hate_samples.csv not found at {hate_samples_path}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None



if __name__ == "__main__":
    print("DATASET CORRESPONDENCE ANALYSIS")
    print("=" * 50)
    
    results = analyze_dataset_correspondence()
    
    if results:
        
        if not results['correspondence_complete']:
            print(f"\n WARNING: Datasets do not have perfect correspondence!")
            print(f"   consistency analysis will show inconsistencies")
            print(f"   even if the data is actually correct within each language.")
    else:
        print("Could not complete analysis. Please check if hate_samples.csv exists.")