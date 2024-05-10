def step_function(num, binary=True):
    if num >= 0:
        return 1
    if binary:
        return 0
    return -1
