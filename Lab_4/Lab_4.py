import random
import math
from numpy import argmin


eps = 0.00001
t_a = 2.326
p0 = 0.999

parts_types_amount = 3
n = int((t_a ** 2) * p0 * (1 - p0) / (eps ** 2))
max_repair_parts = 10

parts_amount_schema = [4, 2, 6]
failure_rates = [40e-6, 10e-6, 80e-6]


def task():
    required_time = 8760
    repair_parts = [0] * parts_types_amount

    for i in range(max_repair_parts):
        repair_parts[0] = i
        for j in range(max_repair_parts):
            repair_parts[1] = j
            for k in range(max_repair_parts):
                repair_parts[2] = k
                p_value = p(required_time, repair_parts)
                if p_value > p0:
                    print(repair_parts, f'P = {p_value}, n = {sum(repair_parts)}')


def p(required_time, repair_parts):
    failures_count = 0
    for _ in range(n):
        time = []
        for i in range(parts_types_amount):
            type_time = []
            for __ in range(parts_amount_schema[i]):
                type_time.append(-1 * math.log(random.random()) / failure_rates[i])
            for __ in range(repair_parts[i]):
                min_time_index = argmin(type_time)
                type_time[min_time_index] = type_time[min_time_index] - math.log(random.random()) / failure_rates[i]
            for j in range(parts_amount_schema[i]):
                time.append(type_time[j])
        if not lfrs_check(time, required_time):
            failures_count += 1
    return 1 - failures_count / n


def lfrs_check(time, required_time):
    return ((time[0] > required_time and time[1] > required_time or time[2] > required_time and time[3] > required_time) and
            (time[4] > required_time and time[5] > required_time) and
            (time[6] > required_time and time[7] > required_time or time[8] > required_time and time[9] > required_time or
             time[10] > required_time and time[11] > required_time))


if __name__ == '__main__':
    task()