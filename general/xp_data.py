import ast
from typing import List, TypeVar, Callable, Type, Dict, Tuple, get_origin
import os
import pandas as pd
from misc import dict_param
from dataclasses import dataclass, is_dataclass, fields, asdict


class XpData:

    def __init__(self, file_path: str, data_cls: Type, values: Dict[str, Type], auto_save: int = -1):
        if not is_dataclass(data_cls):
            raise TypeError(f"{data_cls} needs to be a dataclass")
        self._index_type = data_cls
        self._file_path = file_path
        self._index_cols = [f.name for f in fields(data_cls)]  # TODO: check what is DataclassInstance (in _typeshed?)
        self._values_cols = list(values.keys())
        self.reload()

        self.saved_size = len(self.df)
        self.auto_save = auto_save


    def __len__(self):
        return len(self.df)

    def reload(self):
        if os.path.exists(self._file_path):
            # self.df = pd.read_csv(self._file_path, index_col=self._index_cols)
            self.df = pd.read_csv(self._file_path)
            tuple_columns = [f.name for f in fields(self._index_type) if get_origin(f.type) == tuple]
            for col in tuple_columns:
                self.df[col] = self.df[col].apply(ast.literal_eval)
        else:
            self.df = pd.DataFrame(columns=self._index_cols + self._values_cols)
            # self.df.set_index(self._index_cols, inplace=True)

    def save(self):
        self.df.sort_values(by=self._index_cols, inplace=True)
        self.df.to_csv(self._file_path, index=False)

    # @dict_param
    # def contains_index(self, index_data):
    #     assert isinstance(index_data, self._index_type)
    #     # def contains_index(self, *arg, **index_values):
    #     # if len(arg) > 1:
    #     #     raise Exception('should not have more than 1 argument')
    #     # if len(arg) == 1:
    #     #     if type(arg[0]) != dict:
    #     #         raise Exception('The single argument for this function must be a dictionary')
    #     #     if len(index_values) > 0:
    #     #         raise Exception('If these is a dictionary argument, you cannot pass more named arguments')
    #     #     index_values = arg[0]
    #     # index_tuple = tuple(index_values[col] for col in self.df.index.names)  # Create index tuple
    #     # return index_tuple in self.df.index
    #     return asdict(index_data) in self.df.index

    # @dict_param
    def contains_index(self, index_data):
        assert isinstance(index_data, self._index_type)
        return (self.df[self._index_cols] == pd.Series(asdict(index_data))).all(axis=1).any()

    def add_entry(self, index_data, **entry_values):
        assert isinstance(index_data, self._index_type)
        row = {**asdict(index_data), **entry_values}
        if self.df.empty:
            self.df = pd.DataFrame([row])
        else:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        if self.auto_save > 0 and len(self) >= self.saved_size + self.auto_save:
            self.save()

