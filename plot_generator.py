

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import src.plots_utils as plots_utils
from src.analysis import *

if __name__ == '__main__':
    plots_utils.create_plots_directory('plots')
    plots_utils.create_plots_directory('plots/image')
    plots_utils.create_plots_directory('plots/pdf')

    root_dir = 'dataset/data'
    plots_dir = 'plots'
    change_path(data_dir=root_dir, plots_dir=plots_dir)
    df_annotations, df_llm_responses, df_hate_samples = data_loading()
    df_merged = merge_data(df_annotations, df_llm_responses, df_hate_samples)
    dataset_statistics(df_merged)

    latex_check()

    model_human_agreement(df_annotations)
    safety_analysis(df_merged)
    counter_speech_analysis(df_merged)
    label_distribution_analysis(df_merged)
    inter_model_agreement(df_merged)

    plots_utils.plot_dataset_target_distribution(df_hate_samples, output_root='plots/pdf')
    # plots_utils.plot_language_annotation(df_annotations, df_llm_responses, df_hate_samples, output_root='plots/pdf')

