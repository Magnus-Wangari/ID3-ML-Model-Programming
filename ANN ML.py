import numpy
# import random
# import os

# learning rate
lr = 1

# weights
# weights[w1j, w1i, w2j, w2i, w3j, w3i, Wjk, Wik]
weights = [0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.5, 0.1]

# maximum and minimum values of the data provided
max_value = 70
min_value = 15


# a,b and c are inputs
# d is output
def back(a, b, c, d):
    input1 = ((a - min_value) / (max_value - min_value))
    input2 = ((b - min_value) / (max_value - min_value))
    input3 = ((c - min_value) / (max_value - min_value))
    output = ((d - min_value) / (max_value - min_value))

    # input into node j
    j = (input1 * weights[0]) + (input2 * weights[2]) + (input3 * weights[4])
    output_j = 1 / (1 + numpy.exp(-j))  # sigmoid function
    # input into node i
    i = (input1 * weights[1]) + (input2 * weights[3]) + (input3 * weights[5])
    output_i = 1 / (1 + numpy.exp(-i))  # sigmoid function

    # input of k
    k = (output_j * weights[6]) + (output_i * weights[7])

    # output from k
    output_k = 1 / (1 + numpy.exp(-k))

    if output > 0:
        pass
    else:
        # error value at node k
        error = (output - output_k) * output_k * (1 - output_k)

        # errors for the hidden layers
        error_j = error * weights[6] * output_j * (1 - output_j)
        error_i = error * weights[7] * output_i * (1 - output_i)

        hidden_layer = numpy.array([error_j, error_i])

        print('Updated hidden layers matrix:', hidden_layer)

        # delta weight updates
        # updated weights connected to j
        weights[0] += lr * error * output_j
        weights[2] += lr * error * output_j
        weights[4] += lr * error * output_j

        # print('Updated weights connected to J weights(w1j, w2j, w3j):(', weights[0], weights[2], weights[4], ')')
        # updated weights connected to i
        weights[1] += lr * error * output_i
        weights[3] += lr * error * output_i
        weights[5] += lr * error * output_i
        # print('Updated weights connected to I weights(w1i, w2i, w3i):(', weights[1], weights[3], weights[5], ')')

        weight = numpy.array([[weights[0], weights[2], weights[4]],
                              [weights[1], weights[3], weights[5]]])
        print('Updated Outer Layer Weight Matrix:\n', weight)


def main():
    # a loop for repeating every situation several times
    # learning stage
    for i in range(1):
        back(30, 40, 50, 20)  # Epoch1
        back(40, 50, 20, 15)  # Epoch2
        back(50, 20, 15, 60)  # Epoch3
        back(20, 15, 60, 70)  # Epoch4
        back(15, 60, 70, 50)  # Epoch5
        back(60, 70, 50, 40)  # Epoch6

    # enter values prompt
    print('Enter x: ')
    x = int(input())
    print('Enter y: ')
    y = int(input())
    print('Enter z')
    z = int(input())

    # convert the data entered to a value between o and 1
    # new value = ((original value - minimum value) / (maximum value - minimum value))
    input1 = ((x - min_value) / (max_value - min_value))
    input2 = ((y - min_value) / (max_value - min_value))
    input3 = ((z - min_value) / (max_value - min_value))

    # print('\n x is ', input1, '\n y is', input2, '\n z is', input3)

    # input into node j
    j = (input1 * weights[0]) + (input2 * weights[2]) + (input3 * weights[4])

    # input into node i
    i = (input1 * weights[1]) + (input2 * weights[3]) + (input3 * weights[5])

    # print('\n input into node j', j, '\n input into node i', i)

    output_j = 1 / (1 + numpy.exp(-j))  # sigmoid function
    output_i = 1 / (1 + numpy.exp(-i))  # sigmoid function

    # print('\n output from j', output_j, '\n output from i', output_i)

    # input of k
    k = (output_j * weights[6]) + (output_i * weights[7])

    # output from k
    output_k = 1 / (1 + numpy.exp(-k))

    print('\n output from k', output_k)


if __name__ == '__main__':
    main()
