"""NN lib"""
from dataclasses import dataclass
import math
import random


def sigmoidf(x: float):
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Mat:
    rows: int
    cols: int
    stride: int
    vec: list

    def get_elem_at(self, i, j, stride=None):
        if not stride:
            stride = self.stride
        return self.vec[i*stride + j]

    def set_elem_at(self, i, j, v, stride=None):
        if not stride:
            stride = self.stride
        self.vec[i*stride + j] = v

    def __str__(self):
        _str = "[\n"
        for i in range(self.rows):
            for j in range(self.cols):
                _str += f"  {self.get_elem_at(i, j)}"
            _str += "\n"
        return f"{_str}]\n"


def mat_print(m: Mat, name: str):
    print(f"{name} = {m}")


def new_mat(rows: int, cols: int, fill: float = 0.0) -> Mat:
    return Mat(
        rows,
        cols,
        cols,
        [fill]*cols*rows
    )


def mat_rand(m: Mat, low: float, high: float):
    for i in range(m.rows):
        for j in range(m.cols):
            m.set_elem_at(i, j, random.random()*(high - low) + low)


def mat_dot(a: Mat, b: Mat):
    assert a.cols == b.rows
    n = a.cols
    dst = new_mat(a.rows, b.cols)
    for i in range(dst.rows):
        for j in range(dst.cols):
            v = 0
            for k in range(n):
                v += a.get_elem_at(i, k) * b.get_elem_at(k, j)
            dst.set_elem_at(i, j, v)
    return dst


def mat_sum(dst: Mat, a: Mat):
    assert (dst.rows, dst.cols) == (a.rows, a.cols)
    for i in range(dst.rows):
        for j in range(dst.cols):
            v = dst.get_elem_at(i, j) + a.get_elem_at(i, j)
            dst.set_elem_at(i, j, v)


def mat_sig(m: Mat):
    for i in range(m.rows):
        for j in range(m.cols):
            m.set_elem_at(i, j, sigmoidf(m.get_elem_at(i, j)))


def mat_row(m: Mat, row: int):
    return Mat(
        1,
        m.cols,
        m.stride,
        [m.get_elem_at(row, j) for j in range(m.cols)]
    )


def mat_copy(dst: Mat, src: Mat):
    assert dst.cols == src.cols
    assert dst.rows == src.rows

    for i in range(dst.rows):
        for j in range(dst.cols):
            dst.set_elem_at(i, j, src.get_elem_at(i, j))


def mat_sub(src: Mat, rows: int, cols: int, stride: int):
    vec = []
    for i in range(rows):
        for j in range(cols):
            vec.append(src.get_elem_at(i, j, stride))
    dst = new_mat(rows, cols)
    dst.vec = vec
    return dst

