import functools
from time import time
import types
from typing import TypeVar, Callable

def time_me(func):
    @functools.wraps(func)
    def wrapper(*arg, **kwarg):
        start_time = time()
        result = func(*arg, **kwarg)
        print(f'{func.__name__} took {time()-start_time} seconds')
        return result

    return wrapper

class Timer:
    def __init__(self):
        self.timers = dict()

    def time(self, print_time: bool=True, key:int =0, msg: str = ''):
        current_time = time()
        if print_time:
            last_time = self.timers.get(0, current_time)
            if len(msg)>0:
                print(msg)
            print(f'{key}: {current_time-last_time} sec passed. ')
        self.timers[key] = current_time

def print_progress_bar(iteration, total, prefix ='', suffix ='', decimals = 1, length = 100, fill ='█', printEnd =''):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


R = TypeVar('R')


def dict_param(func: Callable[..., R]) -> Callable[..., R]:
    """
    Transforms a function which gets a dictionary as a single parameter, to be able to get instead named
    parameters. For example, after decorating
    @dict_param
    def foo(d: dict):
        ...

    the following calls will be the same:
        foo(dict(a=1, b=2, c='3'))  =  foo(a=1, b=2, c='3')

    It similarly works for functions of objects, namely:
        foo(self, dict(a=1, b=2, c='3'))  =  foo(self, a=1, b=2, c='3')
    """
    if isinstance(func, types.FunctionType):
        @functools.wraps(func)
        def wrapper(self, *arg, **index_values) -> R:
            if len(arg) > 1:
                raise Exception('should not have more than 1 argument')
            if len(arg) == 1:
                if type(arg[0]) != dict:
                    raise Exception('The single argument for this function must be a dictionary')
                if len(index_values) > 0:
                    raise Exception('If these is a dictionary argument, you cannot pass more named arguments')
                index_values = arg[0]
            return func(self, index_values)
    else:
        @functools.wraps(func)
        def wrapper(*arg, **index_values) -> R:
            if len(arg) > 1:
                raise Exception('should not have more than 1 argument')
            if len(arg) == 1:
                if type(arg[0]) != dict:
                    raise Exception('The single argument for this function must be a dictionary')
                if len(index_values) > 0:
                    raise Exception('If these is a dictionary argument, you cannot pass more named arguments')
                index_values = arg[0]
            return func(index_values)

    return wrapper

