import pickle
import os
import functools
from typing import Optional

record_func = set()

run_all = True

def load_from_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if func in record_func:
            with open(f'func_{func.__name__}.pkl', 'wb') as f:
                pickle.dump((args, kwargs), f)
            record_func.remove(func)
        return result
    return wrapper

def cached(func, path: Optional[str] = None, run_func: bool = False):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        file_path = f'data/cached_{func.__name__}.pkl' if path is None else path

        if run_func or run_all or not os.path.exists(file_path):
            result = func(*args, **kwargs)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(result, f)

            return result

        # Load from cache
        with open(file_path, "rb") as f:
            return pickle.load(f)


    return wrapper


def run_from_cache(func):
    file_path = f'func_{func.__name__}.pkl'
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            args, kwargs = pickle.load(f)
            return func(*args, **kwargs)
    else:
        print(f'No saved parameters for {func.__name__}')

