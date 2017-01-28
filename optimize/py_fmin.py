from scipy.optimize import fmin
import math
import matplotlib.pyplot as plt


def cbf(Xi):
    global count
    count += 1
    f = banana(Xi, math.sqrt(2))
    plt.scatter(count, f)
    plt.pause(0.05)


def banana(X, a):
    return 100*(X[1] - X[0]**2)**2 + (a - X[0])**2


def main():
    a = math.sqrt(2)
    arg = (a, )
    [xopt, fopt, iter, funcalls, warnflag, allvecs] = fmin(
        banana,
        [-1, 1.2],
        args=arg,
        callback=cbf,
        xtol=1e-4,
        ftol=1e-4,
        maxiter=400,
        maxfun=400,
        disp=True,
        retall=True,
        full_output=True)
    for item in allvecs:
        print('%f, %f' % (item[0], item[1]))

if __name__ == '__main__':
    count = 1
    plt.axis([0, 100, 0, 6.5])
    main()
