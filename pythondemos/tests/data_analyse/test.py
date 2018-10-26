from scipy import integrate

def g(x):
    return (x ** 2)

res, err = integrate.quad(g, -1, 1)
print(res)