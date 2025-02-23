import pickle
import os
import functools

record_func = set()

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

def run_from_cache(func):
    file_path = f'func_{func.__name__}.pkl'
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            args, kwargs = pickle.load(f)
            return func(*args, **kwargs)
    else:
        print(f'No saved parameters for {func.__name__}')

@load_from_cache
def say_hello(name: str):
    print(f'hello {name}')

# say_hello('David')

run_from_cache(say_hello)