from tabulate import tabulate


def adaline(inputs, target, weights, b, learning_rate, epochs):
    for i in range(epochs):
        t_err = 0
        print("EPOCH", i + 1)
        rows = [[] for i in range(len(target))]
        for j in range(len(target)):
            rows[j] += [inputs[0][j], inputs[1][j]]
            yi = inputs[0][j] * weights[0] + inputs[1][j] * weights[1] + 1 * b
            dif = target[j] - yi
            dw = [learning_rate * dif * inputs[0][j], learning_rate * dif * inputs[1][j]]
            weights[0] += dw[0]
            weights[1] += dw[1]
            b += learning_rate * dif
            err = dif ** 2
            t_err += err
            rows[j] += [target[j], yi, dw[0], dw[1], learning_rate * dif, weights[0], weights[1], b, err]
        headers = ['x1', 'x2', 'y', 'yi', '△w1', '△w2', '△b', 'w1', 'w2', 'b', '(t-yi)²']
        print(tabulate(rows, headers=headers, tablefmt='fancy_grid'), '\n')
        print("Total Error :", t_err, "\n")
    return weights, b


if __name__ == '__main__':
    inputs = [[1, 1, -1, -1], [1, -1, 1, -1]]
    target = [1, 1, 1, -1]
    weights = [0.1, 0.1]
    b = 0.1
    learning_rate = 0.1

    adaline(inputs, target, weights, b, learning_rate, epochs=5)
