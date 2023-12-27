import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_style('darkgrid')

if __name__ == '__main__':
    df_lipnet_overlapped_name = 'lipnet_overlapped'
    df_vsr_extended_name = 'vsr_extended'

    df_lipnet_overlapped = pd.read_csv('./lipnet_results_overlapped.csv')
    df_vsr_extended = pd.read_csv('./vsr_results.csv')

    # overall_mean = df_lipnet_overlapped[['wer', 'cer']].mean()
    # overall_mean_df = pd.DataFrame(overall_mean).reset_index()
    # overall_mean_df.columns = ['error', 'value']
    # plt.figure(figsize=(10,6))
    # sns.barplot(data=overall_mean_df, x = 'error', y='value', palette='muted', alpha=0.7)
    # plt.title('Среднее значение WER и CER для LipNet overlapped')
    # plt.xlabel('Ошибка', fontsize=16)
    # plt.ylabel('Среднее значение', fontsize=16)
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_means_wercer.png', dpi=400)

    # plt.show()


    # overall_mean = df_vsr_extended[['wer', 'cer']].mean()
    # overall_mean_df = pd.DataFrame(overall_mean).reset_index()
    # overall_mean_df.columns = ['error', 'value']
    # plt.figure(figsize=(10,6))
    # sns.barplot(data=overall_mean_df, x = 'error', y='value', palette='muted', alpha=0.7)
    # plt.title('Среднее значение WER и CER для Auto-VSR на расширенных данных')
    # plt.xlabel('Ошибка', fontsize=16)
    # plt.ylabel('Среднее значение', fontsize=16)
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'./figs/{df_vsr_extended_name}' + '_means_wercer.png', dpi=400)

    # plt.show()

    # overall_mean_vsr = df_vsr_extended[['wer', 'cer']].mean()
    # overall_mean_df_vsr = pd.DataFrame(overall_mean_vsr).reset_index()
    # overall_mean_df_vsr.columns = ['error', 'value']

    # overall_mean = df_lipnet_overlapped[['wer', 'cer']].mean()
    # overall_mean_df = pd.DataFrame(overall_mean).reset_index()
    # overall_mean_df.columns = ['error', 'value']
    # plt.figure(figsize=(10,6))
    # sns.barplot(data=overall_mean_df_vsr, x = 'error', y='value', palette=['#3366FF','#FF5733'], alpha=1)
    # sns.barplot(data=overall_mean_df, x = 'error', y='value', palette=['#FF6EC7','#D462FF'], alpha=0.7)

    # plt.title('Среднее значение WER и CER')
    # plt.xlabel('Ошибка', fontsize=16)
    # plt.ylabel('Среднее значение', fontsize=16)
    # plt.tight_layout()
    # plt.legend()
    # # plt.savefig(f'./figs/{df_vsr_extended_name}' + '_means_wercer.png', dpi=400)

    # plt.show()