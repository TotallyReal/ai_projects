from time import time
import functools


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


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = ''):
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

