import math
import random
from pprint import pprint
from dataclasses import dataclass


XOR_TRAIN = (
    (0, 0, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 0),
)


def sigmoidf(x: float):
    return 1.0 / (1.0 + math.exp(-x))


# along with XOR it can do OR, NAND, AND, NOR
@dataclass
class XOR:
    """
    Modeling (x OR y) AND (x NAND y) -> XOR

        # NN model
        x1 --- > N1
          *   *    *
            *        * -> N3 ---> y
          *   *    *       ^      ^
        x2 --- > N2        |      |
        ^        ^         |      |
      input    layer1    layer2  output
    """
    # neuron 1 (N1)
    or_w1: float = 0
    or_w2: float = 0
    or_b: float = 0

    # neuron 2 (N2)
    nand_w1: float = 0
    nand_w2: float = 0
    nand_b: float = 0

    # neuron 3 (N3)
    and_w1: float = 0
    and_w2: float = 0
    and_b: float = 0

    def forward(self, x1, x2):
        a = sigmoidf(self.or_w1*x1 + self.or_w2*x2 + self.or_b)
        b = sigmoidf(self.nand_w1*x1 + self.nand_w2*x2 + self.nand_b)
        return sigmoidf(self.and_w1*a + self.and_w2*b + self.and_b)

    def cost(self):
        result = 0
        for i in range(len(XOR_TRAIN)):
            x1 = XOR_TRAIN[i][0]
            x2 = XOR_TRAIN[i][1]
            y = self.forward(x1, x2)
            d = y - XOR_TRAIN[i][2]  # error
            result += d * d
        result /= len(XOR_TRAIN)
        return result

    def finite_diff(self, eps: float):
        _xor = XOR()
        c = self.cost()

        saved = self.or_w1
        self.or_w1 += eps
        _xor.or_w1 = (self.cost() - c)/eps
        self.or_w1 = saved

        saved = self.or_w2
        self.or_w2 += eps
        _xor.or_w2 = (self.cost() - c)/eps
        self.or_w2 = saved

        saved = self.or_b
        self.or_b += eps
        _xor.or_b = (self.cost() - c)/eps
        self.or_b = saved

        saved = self.nand_w1
        self.nand_w1 += eps
        _xor.nand_w1 = (self.cost() - c)/eps
        self.nand_w1 = saved

        saved = self.nand_w2
        self.nand_w2 += eps
        _xor.nand_w2 = (self.cost() - c)/eps
        self.nand_w2 = saved

        saved = self.nand_b
        self.nand_b += eps
        _xor.nand_b = (self.cost() - c)/eps
        self.nand_b = saved

        saved = self.and_w1
        self.and_w1 += eps
        _xor.and_w1 = (self.cost() - c)/eps
        self.and_w1 = saved

        saved = self.and_w2
        self.and_w2 += eps
        _xor.and_w2 = (self.cost() - c)/eps
        self.and_w2 = saved

        saved = self.and_b
        self.and_b += eps
        _xor.and_b = (self.cost() - c)/eps
        self.and_b = saved

        return _xor

    def learn(self, g, rate: float):
        self.or_w1 -= rate*g.or_w1
        self.or_w2 -= rate*g.or_w2
        self.or_b -= rate*g.or_b
        self.nand_w1 -= rate*g.nand_w1
        self.nand_w2 -= rate*g.nand_w2
        self.nand_b -= rate*g.nand_b
        self.and_w1 -= rate*g.and_w1
        self.and_w2 -= rate*g.and_w2
        self.and_b -= rate*g.and_b


def rand_xor():
    return XOR(
        or_w1=random.random(),
        or_w2=random.random(),
        or_b=random.random(),
        nand_w1=random.random(),
        nand_w2=random.random(),
        nand_b=random.random(),
        and_w1=random.random(),
        and_w2=random.random(),
        and_b=random.random(),
    )


def main():
    eps = 1e-1
    xor = rand_xor()
    for _ in range(100_000):
        g = xor.finite_diff(eps)
        xor.learn(g, 1e-1)
    print(xor.cost())

    print("-----XOR gate-------")
    for i in range(2):
        for j in range(2):
            print(f"{i} ^ {j} = {xor.forward(i, j)}")

    print("-----OR gate-------")
    for i in range(2):
        for j in range(2):
            print(f"{i} | {j} = {sigmoidf(xor.or_w1*i + xor.or_w2*j + xor.or_b)}")

    print("-----NAND gate-----")
    for i in range(2):
        for j in range(2):
            print(
                f"~({i} & {j}) = {sigmoidf(xor.nand_w1*i + xor.nand_w2*j + xor.nand_b)}")

    print("-----AND gate------")
    for i in range(2):
        for j in range(2):
            print(f"{i} & {j} = {sigmoidf(xor.and_w1*i + xor.and_w2*j + xor.and_b)}")


if __name__ == "__main__":
    main()
