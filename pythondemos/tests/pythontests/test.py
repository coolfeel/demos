import turtle
import math


# turtle.setup(800, 800, 0, 0)
# turtle.color('red')
# turtle.goto(100, 100)
# turtle.goto(100, -100)
# turtle.goto(-100, -100)
# turtle.goto(-100, 100)
# turtle.goto(0, 0)

# turtle.circle(5, 90)
# turtle.circle(10, 90)
# turtle.circle(15, 90)
# turtle.fd(100)


# turtle.left(45)
# turtle.fd(150)
# turtle.right(135)
# turtle.fd(300)
# turtle.right(135)
# turtle.fd(300)


# turtle.done()

# a = [1, 1, 2]
# b = a.count(1)
# print(b)

import numpy as np


num1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
m = np.mean(num1, axis = 0)
res = num1 - m
print(res)

import keras

model = Sequential()
print(model)