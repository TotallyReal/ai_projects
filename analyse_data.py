from xp_data import XpData
from globals import data_output, data_parameters
import pandas as pd
import matplotlib.pyplot as plt

running_data = XpData(file_path='running_data.csv', index_cols=data_parameters, values_cols=data_output)
df = running_data.df
print(df.index.get_level_values('hidden'))

def show_errors(df, hidden_layers):
    df = df[df.index.get_level_values('hidden') == hidden_layers]
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df.index.get_level_values('epochs')-0.1, df.index.get_level_values('batch_size'),
        c=df['train_error_rate'],
        cmap='viridis', s=100)
    scatter = plt.scatter(
        df.index.get_level_values('epochs')+0.1, df.index.get_level_values('batch_size'),
        c=df['test_error_rate'],
        cmap='viridis', s=100)

    # Add a colorbar to indicate the y value
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('error rate')

    # Labels and title
    plt.xlabel('epochs')
    plt.ylabel('batch_size')
    plt.title(f'train \\ test error rate {hidden_layers}')

    # Show the plot

show_errors(df,str(()))
show_errors(df,str((10,)))
show_errors(df,str((10, 10)))
plt.show()