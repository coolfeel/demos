
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('../datasets/mnists', one_hot = True)

print(mnist)

