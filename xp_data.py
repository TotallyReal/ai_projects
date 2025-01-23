from typing import List
import os
import pandas as pd

class XpData:

    def __init__(self, file_path: str, index_cols: List[str], values_cols: List[str]):
        self._file_path = file_path
        self._index_cols = index_cols
        self._values_cols = values_cols
        self.reload()

    def reload(self):
        if os.path.exists(self._file_path):
            self.df = pd.read_csv(self._file_path, index_col=self._index_cols)
        else:
            self.df = pd.DataFrame(columns=self._index_cols + self._values_cols)
            self.df.set_index(self._index_cols, inplace=True)

    def save(self):
        self.df.sort_index(inplace=True)
        self.df.to_csv(self._file_path, index=True)

    def contains(self, *arg, **index_values):
        if len(arg) > 1:
            raise Exception('should not have more than 1 argument')
        if len(arg) == 1:
            if type(arg[0]) != dict:
                raise Exception('The single argument for this function must be a dictionary')
            if len(index_values) > 0:
                raise Exception('If these is a dictionary argument, you cannot pass more named arguments')
            index_values = arg[0]
        index_tuple = tuple(index_values[col] for col in self.df.index.names)  # Create index tuple
        return index_tuple in self.df.index

    def add_entry(self, **entry_values):
        index_values = tuple(entry_values[col] for col in self.df.index.names)  # Extract index values
        row_values = [entry_values[col] for col in self.df.columns]  # Extract column values
        self.df.loc[index_values] = row_values  # Add row to the DataFrame

    def add_new_column(self, col_name:str, default_value, in_index:bool):
        self.df[col_name] = [default_value] * len(self.df)
        if in_index:
            self._index_cols += [col_name]
            self.df.set_index(self._index_cols, inplace=True)
        else:
            self._values_cols += [col_name]

