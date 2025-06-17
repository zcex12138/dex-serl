import numpy as np

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


print_green("test")