import math
import random
from time import time


# OR-gate
TRAIN = (
    (0, 0, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)


def sigmoidf(x: float):
    return 1.0 / (1.0 + math.exp(-x))


# 2 inputs neuron
def cost(w1, w2, b):
    result = 0
    for i in range(len(TRAIN)):
        x1 = TRAIN[i][0]
        x2 = TRAIN[i][1]
        y = sigmoidf(x1 * w1 + x2 * w2 + b)
        d = y - TRAIN[i][2]  # error
        result += d * d
    result /= len(TRAIN)
    return result


def main():
    random.seed(time())
    w1 = random.random()
    w2 = random.random()
    b = random.random()

    eps = 1e-1
    rate = 1e-1

    for _ in range(1_00_000):
        c = cost(w1, w2, b)
        dw1 = (cost(w1 + eps, w2, b) - c) / eps
        dw2 = (cost(w1, w2 + eps, b) - c) / eps
        db = (cost(w1, w2, b + eps) - c) / eps
        w1 -= rate * dw1
        w2 -= rate * dw2
        b -= rate * db

    print(f"c = {cost(w1, w2, b)}, w1 = {w1}, w2 = {w2} b = {b}")

    for i in range(2):
        for j in range(2):
            print(f"{i} | {j} = {sigmoidf(i * w1 +  j* w2 + b)}")


if __name__ == "__main__":
    main()
