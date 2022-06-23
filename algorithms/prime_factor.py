# %%
n = 15

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i != 0: # if there is a remainder increase i by 1 and iterate.
            i += 1
        else:
            n = n // i #integer division
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

# %%
prime_factors(10)