import math
import random
import matplotlib.pyplot as plt
import numpy as np


def calc_math_exp(sequence):
    return sum(sequence)/len(sequence)


def calc_dispersion(sequence, math_exp=None):
    if math_exp is None:
        math_exp = calc_math_exp(sequence)

    return sum([(i - math_exp)**2 for i in sequence])/len(sequence)


def gen_rand_sequence(sequence_len):
    sequence = [random.random() for _ in range(sequence_len)]
    math_exp = calc_math_exp(sequence)
    dispersion = calc_dispersion(sequence, math_exp)
    return sequence, math_exp, dispersion


def K(f, sequence, math_exp):
    n = len(sequence)
    up = sum([(sequence[i] - math_exp) * (sequence[i + f] - math_exp) for i in range(n - f)])
    down = sum([(sequence[i] - math_exp) ** 2 for i in range(n)])
    return up/down


def get_correlation_func_values_for_rand_seq(seq_len):
    rand_sequence, math_exp, _ = gen_rand_sequence(seq_len)
    correlation_func_values = [K(i, rand_sequence, math_exp) for i in range(seq_len)]
    return correlation_func_values


def task_2():
    precision = 5
    for n in [10, 100, 1000, 10000]:
        res = gen_rand_sequence(n)
        math_exp = round(res[1], precision)
        math_exp_diff = round(abs(0.5 - math_exp), 5)
        dispersion = round(res[2], precision)
        dispersion_diff = round(abs(0.08333 - dispersion), 5)
        print(f'n = {n}: \t @M@ = {math_exp} diff = {math_exp_diff} \t @D@ = {dispersion} diff = {dispersion_diff}')


def task_3():
    for n in [10, 100, 1000, 10000]:
        correlation_values = get_correlation_func_values_for_rand_seq(n)
        x_values = [i + 1 for i in range(n)]
        plt.figure()
        plt.xlabel('f')
        plt.ylabel('K(f)')
        plt.bar(x_values, correlation_values, edgecolor='black')
        plt.plot([0, n+1], [0, 0])
        print('done', n)

    plt.show()


def density_plot(rand_sequence):
    n = len(rand_sequence)

    x_theory = np.linspace(0, 1, n)
    y_theory = [1] * n

    bins = math.ceil(1 + math.log2(n))

    plt.figure()
    plt.hist(rand_sequence, bins=bins, density=True, edgecolor='black')  #
    plt.plot(x_theory, y_theory, color='orange')
    plt.legend(['Теоретическая', 'Эмпирическая'])
    plt.suptitle(f'Функции плотности распределения \nn = {n}')


def distribution_plot(rand_sequence):
    n = len(rand_sequence)
    x_theory = np.linspace(0, 1, n)
    bins = math.ceil(1 + math.log2(n))
    gap_width = 1 / bins
    cdfY = [0.0] * (bins + 1)
    for i in range(bins):
        inGap = len([num for num in rand_sequence if i * gap_width <= num < (i + 1) * gap_width])
        cdfY[i + 1] = cdfY[i] + inGap / n

    sorted_numbers = np.sort(rand_sequence)
    y_theory = 1. * np.arange(len(sorted_numbers)) / float(len(sorted_numbers) - 1)
    plt.figure()
    plt.plot(np.linspace(0, 1, bins + 1), cdfY)
    plt.plot(x_theory, y_theory, '--', color='orange')
    plt.legend(['Эмпирическая', 'Теоретическая'])
    plt.suptitle(f'Функции распределения \nn = {n}')


def task_4():
    rand_sequence = gen_rand_sequence(10000)[0]
    density_plot(rand_sequence)
    distribution_plot(rand_sequence)

    plt.show()





if __name__ == '__main__':
    task_2()
    task_3()
    task_4()





