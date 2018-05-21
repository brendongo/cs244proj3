def aspl_lower_bound(r, N):
    def helper(r, N):
        R = N - 1
        j = 1
        while True:
            if R - (r * (r - 1) ** (j - 1)) < 0:
                return R, j
            R -= r * (r - 1) ** (j - 1) 
            j += 1

    R, k = helper(r, N)
    numerator = sum(j * r * ((r - 1) ** (j - 1)) for j in xrange(1, k))
    numerator += k * R
    return float(numerator) / (N - 1)
