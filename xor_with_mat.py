
from dataclasses import dataclass
from nn import *


XOR_TRAIN = [
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,]

TD = Mat(
    4,
    3,
    4,
    XOR_TRAIN
)


@dataclass
class Xor:
    a0: Mat | None = None

    l1w: Mat | None = None
    l1b: Mat | None = None
    l1a: Mat | None = None

    l2w: Mat | None = None
    l2b: Mat | None = None
    l2a: Mat | None = None

    def foward(self):

        # layer 1 activation
        self.l1a = mat_dot(self.a0, self.l1w)
        mat_sum(self.l1a, self.l1b)
        mat_sig(self.l1a)

        # layer 2 activation
        self.l2a = mat_dot(self.l1a, self.l2w)
        mat_sum(self.l2a, self.l2b)
        mat_sig(self.l2a)

    def cost(self, ti: Mat, to: Mat) -> float:
        assert ti.rows == to.rows
        assert to.cols == self.l2a.cols
        n = ti.rows

        c = 0
        for i in range(n):
            x = mat_row(ti, i)
            y = mat_row(to, i)
            # print("input", x, "ouput", y)

            mat_copy(self.a0, x)
            self.foward()

            for j in range(to.cols):
                d = self.l2a.get_elem_at(0, j) - y.get_elem_at(0, j)
                c += d*d
        return c/n


def new_xor():
    m = Xor()
    # input
    m.a0 = new_mat(1, 2)

    # layer 1 w, b
    m.l1w = new_mat(2, 2)
    m.l1b = new_mat(1, 2)
    m.l2a = new_mat(1, 2)

    # layer 2 w, b
    m.l2w = new_mat(2, 1)
    m.l2b = new_mat(1, 1)
    m.l2a = new_mat(1, 1)

    return m


def finite_diff(m: Xor, g: Xor, eps: float, ti: Mat, to: Mat):
    c = m.cost(ti, to)

    for i in range(m.l1w.rows):
        for j in range(m.l1w.cols):
            saved = m.l1w.get_elem_at(i, j)
            m.l1w.set_elem_at(i, j, saved+eps)
            g.l1w.set_elem_at(i, j, (m.cost(ti, to) - c)/eps)
            m.l1w.set_elem_at(i, j, saved)

    for i in range(m.l1b.rows):
        for j in range(m.l1b.cols):
            saved = m.l1b.get_elem_at(i, j)
            m.l1b.set_elem_at(i, j, saved+eps)
            g.l1b.set_elem_at(i, j, (m.cost(ti, to) - c)/eps)
            m.l1b.set_elem_at(i, j, saved)

    for i in range(m.l2w.rows):
        for j in range(m.l2w.cols):
            saved = m.l2w.get_elem_at(i, j)
            m.l2w.set_elem_at(i, j, saved+eps)
            g.l2w.set_elem_at(i, j, (m.cost(ti, to) - c)/eps)
            m.l2w.set_elem_at(i, j, saved)

    for i in range(m.l2b.rows):
        for j in range(m.l2b.cols):
            saved = m.l2b.get_elem_at(i, j)
            m.l2b.set_elem_at(i, j, saved+eps)
            g.l2b.set_elem_at(i, j, (m.cost(ti, to) - c)/eps)
            m.l2b.set_elem_at(i, j, saved)


def learn(m: Xor, g: Xor, rate: float):
    for i in range(m.l1w.rows):
        for j in range(m.l1w.cols):
            m1 = m.l1w.get_elem_at(i, j)
            g1 = g.l1w.get_elem_at(i, j)
            m.l1w.set_elem_at(i, j, m1 - rate*g1)

    for i in range(m.l1b.rows):
        for j in range(m.l1b.cols):
            m1 = m.l1b.get_elem_at(i, j)
            g1 = g.l1b.get_elem_at(i, j)
            m.l1b.set_elem_at(i, j, m1 - rate*g1)

    for i in range(m.l2w.rows):
        for j in range(m.l2w.cols):
            m1 = m.l2w.get_elem_at(i, j)
            g1 = g.l2w.get_elem_at(i, j)
            m.l2w.set_elem_at(i, j, m1 - rate*g1)

    for i in range(m.l2b.rows):
        for j in range(m.l2b.cols):
            m1 = m.l2b.get_elem_at(i, j)
            g1 = g.l2b.get_elem_at(i, j)
            m.l2b.set_elem_at(i, j, m1 - rate*g1)


def main():

    ti = mat_sub(TD, TD.rows, 2, 3)
    TD.vec = TD.vec[2:]
    to = mat_sub(TD, TD.rows, 1, 3)

    m = new_xor()
    g = new_xor()

    mat_rand(m.l1w, 0, 1)
    mat_rand(m.l1b, 0, 1)
    mat_rand(m.l2w, 0, 1)
    mat_rand(m.l2b, 0, 1)

    eps = 1e-1
    rate = 1e+1

    for _ in range(10_000):
        finite_diff(m, g, eps, ti, to)
        learn(m, g, rate)
    print("cost =", m.cost(ti, to))

    print("----------------------")
    for i in range(2):
        for j in range(2):
            m.a0.set_elem_at(0, 0, i)
            m.a0.set_elem_at(0, 1, j)
            m.foward()
            y = m.l2a.vec[0]

            print(f"{i} ^ {j} = {y}")


if __name__ == "__main__":
    main()
