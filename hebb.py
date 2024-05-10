from tabulate import tabulate
from activation_functions import *


def hebb(inputs, target):
    weights = [0] * len(inputs)
    bias = 0
    rows = [[] for i in range(len(target))]
    dw = []
    for i in range(len(target)):
        for j in range(len(inputs)):
            rows[i] += [inputs[j][i]]
            dw += [inputs[j][i] * target[i]]
            weights[j] += dw[j]
        bias += target[i]

        # for printing table
        rows[i] += [1, target[i]]
        rows[i] += [dw[j] for j in range(len(dw))]
        rows[i] += [target[i]]
        rows[i] += [weights[j] for j in range(len(weights))]
        rows[i] += [bias]
        dw = []
    # for printing table
    headers = [f'x{i + 1}' for i in range(len(inputs))]
    headers += ['b', 'y']
    headers += [f'△w{i + 1}' for i in range(len(inputs))]
    headers += ['△b']
    headers += [f'w{i + 1}' for i in range(len(inputs))]
    headers += ['b']
    print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))
    return weights, bias


def test(inputs, target, weights, bias, is_binary):
    print('Test:')
    for i in range(len(target)):
        s = 0
        for j in range(len(inputs)):
            s += weights[j] * inputs[j][i]
            print(f'{inputs[j][i]},', end='')
        s += bias
        print(f' y={step_function(s, binary=is_binary)}')
        if step_function(s, binary=is_binary) != target[i]:
            print('Hebb\'s rule is not valid')
            return
    print('Hebb\'s rule is valid')


if __name__ == '__main__':
    # inputs = [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]]
    inputs = [[1, 1, 1, 1, -1, -1, -1, -1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, -1]]
    target = [1, -1, -1, -1, -1, -1, -1, -1]

    w, b = hebb(inputs, target)

    print('\nDecision boundary:')
    for i in range(len(w)):
        print(f'({w[i]}) * x{i + 1}', end=' + ')
    print(f'({b}) * b')
    print()
    test(inputs, target, w, b, is_binary=False)
