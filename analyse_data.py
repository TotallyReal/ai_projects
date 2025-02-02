from xp_data import XpData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

running_data = XpData(file_path='running_data2.csv', index_cols=data_parameters, values_cols=data_output)
df = running_data.df


def filter(df, partial_data) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in partial_data.items():
        mask &= (df[col] == value)
    return df[mask]



def show_errors(df, hidden_layers):
    # df = df[df.index.get_level_values('hidden') == hidden_layers]
    # df = df[df['hidden'] == hidden_layers]
    df = df.xs(key=(hidden_layers), level=('hidden'))

    combined_colors = np.concatenate([df['train_error_rate'], df['test_error_rate']])
    norm = plt.Normalize(vmin=combined_colors.min(), vmax=combined_colors.max())

    plt.scatter(
        df.index.get_level_values('epochs')-0.1, df.index.get_level_values('batch_size'),
        c=df['train_error_rate'],
        cmap='viridis', s=100, norm=norm)
    scatter = plt.scatter(
        df.index.get_level_values('epochs')+0.1, df.index.get_level_values('batch_size'),
        c=df['test_error_rate'],
        cmap='viridis', s=100, norm=norm)

    # Add a colorbar to indicate the y value
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('error rate')

    # Labels and title
    plt.xlim(0, df.index.get_level_values('epochs').max()+1)
    plt.xlabel('epochs')
    plt.ylabel('batch_size')
    plt.title(f'train \\ test error rate {hidden_layers}')

    # Show the plot

plt.figure(figsize=(12, 9))
# for k in range(6):
#     plt.subplot(2, 3, k+1)
#     show_errors(df,str((k+1,)))
# plt.show()


df = filter(df,{'batch_size':64, 'hidden':str((2,2,2))})
print(df)