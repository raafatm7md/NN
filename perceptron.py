from tabulate import tabulate


def activation(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def perceptron(inputs, target, learning_rate):
    weights = [0] * len(inputs)
    bias = 0
    for epoch in range(1000):
        rows = [[] for i in range(len(target))]
        ty = []
        for i in range(len(target)):
            dw = []
            y_in = bias
            for j in range(len(inputs)):
                rows[i] += [inputs[j][i]]
                y_in += inputs[j][i]*weights[j]
            y = activation(y_in)
            ty.append(y)
            if y != target[i]:
                for j in range(len(inputs)):
                    dw += [learning_rate * inputs[j][i] * target[i]]
                    weights[j] += dw[j]
                db = learning_rate * target[i]
                bias += db
            else:
                dw = [0] * len(inputs)
                db = 0
            rows[i] += [target[i], y_in, y]+[dw[j] for j in range(len(dw))]+[db]+[weights[j] for j in range(len(weights))]+[bias]
            dw = []
        headers = [f'x{i + 1}' for i in range(len(inputs))]+['y', 'y_in', 'ŷ']+[f'△w{i + 1}' for i in range(len(inputs))]+['△b']+[f'w{i + 1}' for i in range(len(inputs))]+['b']
        print(f'EPOCH {epoch+1}:')
        print(tabulate(rows, headers=headers, tablefmt='fancy_grid'), '\n')
        if target == ty: break
    return weights, bias


if __name__ == '__main__':
    inputs = [[1, 1, 0, 0], [1, 0, 1, 0]]
    # inputs = [[1, 1, -1, -1], [1, -1, 1, -1]]
    target = [1, -1, -1, -1]
    learning_rate = 1
    w, b = perceptron(inputs, target, learning_rate)

    print('\nDecision boundary:\ny = ', end='')
    for i in range(len(w)):
        print(f'({w[i]}) * x{i + 1}', end=' + ')
    print(f'({b})')
    print()
