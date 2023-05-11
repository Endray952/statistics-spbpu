import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as theoretical

# constants
sample_size = 10000
y_pdf_label = 'f(x)'
y_cdf_label = 'F(x)'
pdf_title = 'Функция плотности распределения'
cdf_title = 'Функция распределения'
legend_label = ['Теоретическая', 'Эмпирическая']
x_label = 'x'


def get_distribution_properties(values):
    math_exp = sum(values) / len(values)
    dispersion = sum([x ** 2 for x in values]) / len(values) - math_exp ** 2
    return [math_exp, dispersion]


def print_distribution_results(m_th, d_th, m_emp, d_emp, precision=5):
    print(f'\nТеоретические:\tM = {round(m_th, precision)} \tD = {round(d_th, precision)}')
    print(f'Эмпирические:\tM = {round(m_emp, precision)} \tD = {round(d_emp, precision)}')
    print(f'Погрешность:\tdM = {round(math.fabs(m_emp - m_th), precision)} \tdD = {round(math.fabs(d_emp - d_th), precision)}\n')


def uniform_distribution(a, b):
    return (b - a) * random.random() + a


def normal_distribution_clt(m, s):
    return m + s * (sum([random.random() for _ in range(12)]) - 6)


def normal_distribution_box_miller(m, s):
    return m + s * math.sqrt(-2 * math.log(random.random(), math.e)) * math.cos(2 * math.pi * random.random())


def exponential_distribution(beta):
    return -beta * math.log(random.random(), math.e)


def chi2_distribution(n):
    return sum([normal_distribution_box_miller(0, 1) ** 2 for _ in range(n)])


def student_distribution(n):
    return normal_distribution_box_miller(0, 1) / math.sqrt(chi2_distribution(n) / n)


def erlang_distribution(beta, k):
    return sum([exponential_distribution(beta) for _ in range(k)])


def task_1():
    a = -3
    b = 9
    values = [uniform_distribution(a, b) for _ in range(sample_size)]

    m_emp, d_emp = get_distribution_properties(values)
    print_distribution_results((a + b) / 2, ((b - a) ** 2) / 12, m_emp, d_emp)

    plt.figure()
    plt.plot([a, b], [(1 / (b - a)), (1 / (b - a))], color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.title(pdf_title)
    plt.legend(legend_label)

    plt.figure()
    plt.plot([a, b], [0, 1], color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.title(cdf_title)
    plt.legend(legend_label)

    plt.show()



def task_2():
    m = 0
    s = 1

    def clt():
        values = [normal_distribution_clt(m,s) for _ in range(sample_size)]
        m_emp, d_emp = get_distribution_properties(values)
        print('central limit theorem')
        print_distribution_results(m, s ** 2, m_emp, d_emp)
        x_th = np.linspace(min(values), max(values), num=100)
        y_th_pdf = theoretical.norm.pdf(x_th, m, s)
        plt.figure()
        plt.plot(x_th, y_th_pdf, color='orange')
        plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_pdf_label)
        plt.title(pdf_title)
        plt.legend(legend_label)

        y_th_cdf = theoretical.norm.cdf(x_th, m, s)
        plt.figure()
        plt.plot(x_th, y_th_cdf, color='orange')
        plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_cdf_label)
        plt.title(cdf_title)
        plt.legend(legend_label)

        plt.show()

    def box_miller():
        values = [normal_distribution_box_miller(m, s) for _ in range(sample_size)]
        m_emp, d_emp = get_distribution_properties(values)
        print('Box-Miller')
        print_distribution_results(m, s ** 2, m_emp, d_emp)
        x_th = np.linspace(min(values), max(values), num=100)
        y_th_pdf = theoretical.norm.pdf(x_th, m, s)

        plt.figure()
        plt.plot(x_th, y_th_pdf, color='orange')
        plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_pdf_label)
        plt.title(pdf_title)
        plt.legend(legend_label)

        y_th_cdf = theoretical.norm.cdf(x_th, m, s)
        plt.figure()
        plt.plot(x_th, y_th_cdf, color='orange')
        plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
        plt.xlabel(x_label)
        plt.ylabel(y_cdf_label)
        plt.title(cdf_title)
        plt.legend(legend_label)

        plt.show()

    clt()
    box_miller()


def task_3():
    beta = 1
    values = [exponential_distribution(beta) for _ in range(sample_size)]

    m_emp, d_emp = get_distribution_properties(values)
    print_distribution_results(beta, beta ** 2, m_emp, d_emp)
    x_th = np.linspace(min(values), max(values), num=100)
    y_th_pdf = theoretical.expon.pdf(x_th, scale=beta)
    plt.figure()
    plt.plot(x_th, y_th_pdf, color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.title(pdf_title)
    plt.legend(legend_label)

    y_th_cdf = theoretical.expon.cdf(x_th, scale=beta)
    plt.figure()
    plt.plot(x_th, y_th_cdf, color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.title(cdf_title)
    plt.legend(legend_label)

    plt.show()


def task_4():
    n = 10
    values = [chi2_distribution(n) for _ in range(sample_size)]

    m_emp, d_emp = get_distribution_properties(values)
    print_distribution_results(n, 2 * n, m_emp, d_emp)

    x_th = np.linspace(min(values), max(values), num=100)
    y_th_pdf = theoretical.chi2.pdf(x_th, n)
    plt.figure()
    plt.plot(x_th, y_th_pdf, color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.title(pdf_title)
    plt.legend(legend_label)

    y_th_cdf = theoretical.chi2.cdf(x_th, n)
    plt.figure()
    plt.plot(x_th, y_th_cdf, color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.title(cdf_title)
    plt.legend(legend_label)

    plt.show()


def task_5():
    n = 10
    values = [student_distribution(n) for _ in range(sample_size)]

    m_emp, d_emp = get_distribution_properties(values)
    print_distribution_results(0, n / (n - 2), m_emp, d_emp)

    x_th = np.linspace(min(values), max(values), num=100)
    y_th_pdf = theoretical.t.pdf(x_th, n)
    plt.figure()
    plt.plot(x_th, y_th_pdf, color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.title(pdf_title)
    plt.legend(legend_label)

    y_th_cdf = theoretical.t.cdf(x_th, n)
    plt.figure()
    plt.plot(x_th, y_th_cdf, color='orange')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.title(cdf_title)
    plt.legend(legend_label)

    plt.show()


def individual_task():
    def erlang_cdf(x):
        return math.exp(-x) * (-(x ** 3) - 3 * (x ** 2) - 6 * x - 6) / 6 + 1

    amount = 100
    beta = 1
    k = 4

    values = sorted([erlang_distribution(beta, k) for _ in range(amount)])

    max_diff = max([math.fabs((r + 1) / amount - erlang_cdf(values[r])) for r in range(len(values))])

    m_emp, d_emp = get_distribution_properties(values)
    print_distribution_results(k / beta, k / beta ** 2, m_emp, d_emp)
    print(f'Максимальная разность:\t{round(max_diff, 2)}')
    print(f'Критическое значение: \t{0.136}')

    x_th = np.linspace(min(values), max(values), num=100)

    y_th_pdf = theoretical.erlang.pdf(x_th, k)
    plt.figure()
    plt.plot(x_th, y_th_pdf, color='red')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.title(pdf_title)
    plt.legend(legend_label)

    y_th_cdf = theoretical.erlang.cdf(x_th, k)
    plt.figure()
    plt.plot(x_th, y_th_cdf, color='red')
    plt.hist(values, bins=math.ceil(1 + math.log2(len(values))), density=True, cumulative=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.title(cdf_title)
    plt.legend(legend_label)

    plt.show()




if __name__ == '__main__':
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    individual_task()
