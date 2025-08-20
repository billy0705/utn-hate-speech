from src.analysis import *

def print_stats(df):
    df_human = df[df["Human_Label"].notna()]

    # Count per Language and per Human_Label
    stats = df_human["Language"].value_counts()

    print("\nStats per Language:\n", stats, "\n")

if __name__ == '__main__':

    root_dir = 'dataset/data'
    plots_dir = 'plots'
    change_path(data_dir=root_dir, plots_dir=plots_dir)
    df_annotations, df_llm_responses, df_hate_samples = data_loading()
    df_merged = merge_data(df_annotations, df_llm_responses, df_hate_samples)
    print(df_merged.info())

    print("Please select the language you want to annotate: 1. English 2. Arabic 3. Albanian 4. Chinese\n")
    language_choice = input("Enter the number corresponding to your choice: ")

    selected_language = {
        "1": "English",
        "2": "Arabic",
        "3": "Albanian",
        "4": "Chinese"
    }.get(language_choice, "English")

    print(f"You have selected {selected_language} for annotation.")

    # Proceed with the annotation process for the selected language

    df_merged_filtered = df_merged[df_merged['Language'] == selected_language]

    df_random = df_merged_filtered.sample(frac=1, random_state=None).reset_index(drop=True)

    i = 0
    shows_status = False
    while i < len(df_random):
        if shows_status:
            print_stats(df_merged)
            user_input = input(
                "Enter the following numbers or actions \n"
                "\t5: Continue\n"
                "\t6: Save annotations\n"
                "\t7: Status for language counts\n"
                "\t8: Exit & save\n"
                "Your input: "
            )
        else:
            response_id = df_random.iloc[i]['Response_ID']
            print(f"\n\nAnnotating response ID: {response_id}\n")
            print(f"Hate speech:\n{df_random.iloc[i].Text}\n")
            print(f"LLM response:\n{df_random.iloc[i].Model_Response}\n")

            user_input = input(
                "Enter your annotation in the following numbers or actions \n"
                "\t1: Hate speech\n"
                "\t2: Neutral\n"
                "\t3: Refusal\n"
                "\t4: Counter-Speech\n"
                "\t6: Save annotations\n"
                "\t7: Status for language counts\n"
                "\t8: Exit & save\n"
                "\t9: Exit & w/o save\n"
                "Your input: "
            )

        if user_input == '1' and shows_status is False:
            df_annotations.loc[df_annotations['Response_ID'] == response_id, 'Human_Label'] = 'Hate speech'
            df_merged.loc[df_merged['Response_ID'] == response_id, 'Human_Label'] = 'Hate speech'
            i += 1
        elif user_input == '2' and shows_status is False:
            df_annotations.loc[df_annotations['Response_ID'] == response_id, 'Human_Label'] = 'Neutral'
            df_merged.loc[df_merged['Response_ID'] == response_id, 'Human_Label'] = 'Neutral'
            i += 1
        elif user_input == '3' and shows_status is False:
            df_annotations.loc[df_annotations['Response_ID'] == response_id, 'Human_Label'] = 'Refusal'
            df_merged.loc[df_merged['Response_ID'] == response_id, 'Human_Label'] = 'Refusal'
            i += 1
        elif user_input == '4' and shows_status is False:
            df_annotations.loc[df_annotations['Response_ID'] == response_id, 'Human_Label'] = 'Counter-Speech'
            df_merged.loc[df_merged['Response_ID'] == response_id, 'Human_Label'] = 'Counter-Speech'
            i += 1
        elif user_input == '5':
            shows_status = False
            continue
        elif user_input == '6':
            df_annotations.to_csv("dataset/data/annotations.csv", index=False)
            print("Annotations saved.")
            shows_status = False
        elif user_input == '7':
            shows_status = True
        elif user_input == '8':
            df_annotations.to_csv("dataset/data/annotations.csv", index=False)
            print("Annotations saved.")
            break
        elif user_input == '9':
            break
        else:
            print("Invalid input. Please try again.")
    