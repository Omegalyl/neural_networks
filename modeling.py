import random
# doudble
TRAIN = {
    0 : 0,
    1 : 1,
    2 : 4,
    3 : 6,
    4 : 8
}


def cost(w, b):
    result = 0
    for x in TRAIN:
        y = x*w + b
        d = y - TRAIN[x] # error
        result += d*d
    result /= len(TRAIN)
    return result


def main():
    random.seed(69)
    w  = random.random()*10
    b = random.random()*5

    eps = 1e-3
    rate = 1e-1

    for _  in range(500):
        c = cost(w, b)
        dw = (cost(w + eps, b) - c)/eps
        db = (cost(w, b + eps) - c)/eps
        w -= rate*dw
        b -= rate*db
        print(f"c = {cost(w, b)}, w = {w}, b = {b}")



if __name__ == "__main__":
    main()