import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_style('white')
sns.set (font_scale= 1 )
# sns.color_palette("Paired")

if __name__ == '__main__':
    df_lipnet_overlapped_name = 'lipnet_overlapped'
    df_vsr_extended_name = 'vsr_extended'

    df_lipnet_overlapped = pd.read_csv('./lipnet_results_overlapped.csv')
    df_vsr_extended = pd.read_csv('./vsr_results.csv')

    # MEAN WER CER LIPNET OVERLAPPED
    # overall_mean = df_lipnet_overlapped[['wer', 'cer']].mean()
    # overall_mean_df = pd.DataFrame(overall_mean).reset_index()
    # overall_mean_df.columns = ['error', 'value']
    # plt.figure(figsize=(10,6))
    # sns.barplot(data=overall_mean_df, x = 'error', y='value', palette='muted', alpha=0.7)
    # # plt.title('Среднее значение WER и CER для LipNet overlapped')
    # plt.xlabel('Ошибка', fontsize=16)
    # plt.ylabel('Среднее значение', fontsize=16)
    # plt.tight_layout()
    # # plt.legend()
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_means_wercer_wotitle.png', dpi=400)

    # plt.show()

    #  MEAN WER CER FOR AUTO VSR
    # overall_mean = df_vsr_extended[['wer', 'cer']].mean()
    # overall_mean_df = pd.DataFrame(overall_mean).reset_index()
    # overall_mean_df.columns = ['error', 'value']
    # plt.figure(figsize=(10,6))
    # sns.barplot(data=overall_mean_df, x = 'error', y='value', palette='muted', alpha=0.7)
    # # plt.title('Среднее значение WER и CER для Auto-VSR на расширенных данных')
    # plt.xlabel('Ошибка', fontsize=16)
    # plt.ylabel('Среднее значение', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(f'./figs/{df_vsr_extended_name}' + '_means_wercer_wotitle.png', dpi=400)

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

    # WER FROM LEN WORDS LIPNET OVERLAPPED
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x='len word truth', y='wer', color='blue', data=df_lipnet_overlapped)
    # # plt.title('Зависимость WER от длины слов в истинной аннотации')
    # plt.xlabel('Длина истинной транскрипции в словах', fontsize=16)
    # plt.ylabel('WER', fontsize=16)
    # # plt.tight_layout()
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_wer_from_lenwords.png', dpi=400)

    # plt.show()

    # # CER FROM LEN CHARS LIPNET OVERLAPPED
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x='len char truth', y='cer', color='red', data=df_lipnet_overlapped)
    # # plt.title('Зависимость WER от длины слов в истинной аннотации')
    # plt.xlabel('Длина истинной транскрипции в символах', fontsize=16)
    # plt.ylabel('CER', fontsize=16)
    # # plt.tight_layout()
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_cer_from_lenchars.png', dpi=400)

    # plt.show()

    # WER FROM LEN WORDS vsr
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x='len word truth', y='wer', color='green', data=df_vsr_extended)
    # # plt.title('Зависимость WER от длины слов в истинной аннотации')
    # plt.xlabel('Длина истинной транскрипции в словах', fontsize=16)
    # plt.ylabel('WER', fontsize=16)
    # # plt.tight_layout()
    # plt.savefig(f'./figs/{df_vsr_extended_name}' + '_wer_from_lenwords.png', dpi=400)

    # plt.show()

    # CER FROM LEN CHARS vsr
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x='len char truth', y='cer', color='orange', data=df_vsr_extended)
    # # plt.title('Зависимость WER от длины слов в истинной аннотации')
    # plt.xlabel('Длина истинной транскрипции в символах', fontsize=16)
    # plt.ylabel('CER', fontsize=16)
    # # plt.tight_layout()
    # plt.savefig(f'./figs/{df_vsr_extended_name}' + '_cer_from_lenchars.png', dpi=400)

    # plt.show()

    # LEN PRED CHARS VS LEN TRUTH LIPNET
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x='len char truth', y='len char predicted', data=df_lipnet_overlapped, color='red', s=200, marker='D')
    # plt.xlabel('Длина истинной аннотации в символах', fontsize=14)
    # plt.ylabel('Длина предсказанной аннотации в символах', fontsize=14)
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_len_pred_vs_len_truth.png', dpi=400)

    # plt.show()

    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x='len word truth', y='len word predicted', data=df_lipnet_overlapped, color='blue', s=200, marker='^')
    # plt.xlabel('Длина истинной аннотации в словах', fontsize=14)
    # plt.ylabel('Длина предсказанной аннотации в словах', fontsize=14)
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_len_pred_vs_len_truth_words.png', dpi=400)

    # plt.show()

    # LEN PRED CHARS VS LEN TRUTH VSR
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='len char truth', y='len char predicted', data=df_vsr_extended, color='#fe981e', s=200, marker='D')
    plt.xlabel('Длина истинной аннотации в символах', fontsize=14)
    plt.ylabel('Длина предсказанной аннотации в символах', fontsize=14)
    plt.savefig(f'./figs/{df_vsr_extended_name}' + '_len_pred_vs_len_truth1.svg', format='svg',dpi=400)

    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='len word truth', y='len word predicted', data=df_vsr_extended, color='#7030f3', s=200, marker='^')
    plt.xlabel('Длина истинной аннотации в словах', fontsize=14)
    plt.ylabel('Длина предсказанной аннотации в словах', fontsize=14)
    plt.savefig(f'./figs/{df_vsr_extended_name}' + '_len_pred_vs_len_truth_words1.svg', format='svg',dpi=400)

    plt.show()


    # plt.figure(figsize=(4, 3))
    # sns.pairplot(df_vsr_extended)
    # plt.suptitle('Pairplot для взаимосвязей между переменными', y=1.02)
    # plt.savefig(f'./figs/{df_vsr_extended_name}' + '_pairplots.png', dpi=400)

    # plt.show()

    # plt.figure(figsize=(4, 3))
    # sns.pairplot(df_lipnet_overlapped.drop(columns=['video name']))
    # plt.suptitle('Pairplot для взаимосвязей между переменными', y=1.02)
    # plt.savefig(f'./figs/{df_lipnet_overlapped_name}' + '_pairplots.png', dpi=400)

    # plt.show()
