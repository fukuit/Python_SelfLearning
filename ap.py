"""affinity propagation
教師なし学習型のクラスタリング手法
"""
import re
import numpy as np
import sys

def apclust(similarity, preference, output, maxits=500, convits=50, lam=0.5):
    # Read similarities and preferences
    i = []
    k = []
    s = []
    m = 0
    with open(similarity) as simf:
        for line in simf:
            columns = re.split('\s+', line.rstrip())
            if len(columns) == 3:
                i.append(int(columns[0]) - 1)
                k.append(int(columns[1]) - 1)
                s.append(float(columns[2]))
                m += 1
    n = 0
    with open(preference) as pref:
        for line in pref:
            s.append(float(line.rstrip()))
            i.append(n)
            k.append(n)
            n += 1
    m = m + n

    # Initialize availabilities to 0 and run affinity propagation
    a = [0.0] * m
    r = [0.0] * m
    decit = convits
    dec = []
    decsum = []
    mx1 = [-1 * 1.79769313486232e+308] * n
    mx2 = [-1 * 1.79769313486232e+308] * n
    srp = [0.0] * n
    K = 0
    dn = 0
    it = 0
    conv = 0
    while(dn == 0):
        it += 1  # Increase iteration IndexError
        # Compute responsibility
        mx1 = [-1 * 1.79769313486232e+308] * n
        mx2 = [-1 * 1.79769313486232e+308] * n

        for j in range(0, m):

            tmp = a[j] + s[j]
            if tmp > mx1[i[j]]:
                mx2[i[j]] = mx1[i[j]]
                mx1[i[j]] = float(a[j]) + float(s[j])
            elif tmp > mx2[i[j]]:
                mx2[i[j]] = tmp
        for j in range(0, m):
            if float(a[j]) + float(s[j]) == mx1[i[j]]:
                r[j] = lam * r[j] + (1 - lam) * (float(s[j]) - mx2[i[j]])
            else:
                r[j] = lam * r[j] + (1 - lam) * (float(s[j]) - mx1[i[j]])

        # Compute availabilities
        srp = []
        for j in range(0, m-n):
            srp.append(0.0)

        for j in range(0, m-n):
            if r[j] > 0.0:
                srp[k[j]] += r[j]
        for j in range(m-n, m):
            srp[k[j]] += r[j]
        for j in range(0, m-n):
            tmp = srp[k[j]]
            if r[j] > 0.0:
                tmp -= r[j]
            if tmp >= 0.0:
                tmp = 0
            a[j] = lam * a[j] + (1 - lam)*tmp
        for j in range(m-n, m):
            a[j] = lam * a[j] + (1 - lam)*(srp[k[j]] - r[j])
        # Identify exemplars and check to see if finished
        decit += 1
        if decit >= convits:
            decit = 0
        K = 0
        dec = np.zeros((m, m))
        decsum = np.zeros(n)
        for j in range(0, n):
            decsum[j] -= dec[decit, j]
            if a[m - n + j] + r[m - n + j] > 0.0:
                dec[decit, j] = 1
            else:
                dec[decit, j] = 0
            decsum[j] += dec[decit, j]
            K += dec[decit, j]
        if it > convits or it >= maxits:
            # Check convergence
            conv = 1
            for j in range(0, n):
                if decsum[j] != 0 and decsum[j] != convits:
                    conv = 0
            # Check to see if done
            if (conv == 1 and K > 0) or it == maxits:
                dn = 1

    # If clusters were identified, find the assignments and output them
    if K > 0:
        for j in range(0, m):
            if dec[decit, k[j]] == 1:
                a[j] = 0.0
            else:
                a[j] = -1 * 1.79769313486232e+308
        idx = np.zeros(n)
        mx1 = [-1 * 1.79769313486232e+308] * n
        for j in range(0, m):
            tmp = float(a[j]) + float(s[j])
            if tmp > mx1[i[j]]:
                mx1[i[j]] = tmp
                idx[i[j]] = k[j]
        for j in range(0, n):
            if dec[decit, j] != 0:
                idx[j] = j
        for j in range(0, n):
            srp.append(0.0)
        for j in range(0, m):
            if idx[i[j]] == idx[k[j]]:
                srp[k[j]] += float(s[j])
        mx1 = [-1 * 1.79769313486232e+308] * n
        for j in range(0, m):
            tmp = float(a[j]) + float(s[j])
            if tmp > mx1[i[j]]:
                mx1[i[j]] = tmp
                idx[i[j]] = k[j]
        for j in range(0, n):
            if dec[decit, j] != 0:
                idx[j] = j

        with open(output, mode="w", encoding="utf-8") as outf:
            for j in range(0, n):
                outf.write("%d\n" % (idx[j]+1))

        dpsim = 0.0
        expref = 0.0
        for j in range(0, m):
            if idx[i[j]] == k[j]:
                if i[j] == k[j]:
                    expref += float(s[j])
                else:
                    dpsim += float(s[j])
        netsim = dpsim + expref
        print("\nNumber of identified clusters:%d\n" % K)
        print("Fitness (net similarity): %f\n" % netsim)
        print("  Similarities of data points to exemplars: %f\n" % dpsim)
        print("  Preferences of selected exemplars: %f\n" % expref)
        print("Number of iterations: %d\n\n" % it)
    else:
        print("\nDid not identify any clusters\n")
    if conv == 0:
        print("\n*** Warning: Algorithm did not converge. Consider increasing\n")
        print("    maxits to enable more iterations. It may also be necessary\n")
        print("    to increase damping (increase dampfact).\n\n")


if __name__ == "__main__":
    argvs = sys.argv
    argc = len(argvs)
    sim = "Similarities.txt"
    pre = "Preferences.txt"
    out = "index.out.txt"
    if argc == 4:
        sim = argvs[1]
        pre = argvs[2]
        out = argvs[3]
    apclust(sim, pre, out, maxits=2000, convits=100, lam=0.9)
