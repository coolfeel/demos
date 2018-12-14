
import tensorflow as tf


#为了建模方便，tf会将常量转化为1个永远输出固定值的运算
#a = tf.constant([1.0, 2.0], name = 'a')
#输出的第1个参数是节点的名称以及当前张量来自节点的第几个输出
#print(a)
#查看张量所属的计算图,如果没默认指明，则属于默认的计算图
# print(a.graph)
#与默认的计算图对比
# print(tf.get_default_graph())


#定义计算图g1,计算图不仅可以用来隔离张量和计算，还提供了管理张量和计算的机制
g1 = tf.Graph()
#在这个图中操作，定义为‘默认’图
with g1.as_default():
    #在计算图g1中定义变量v,并设置初始为0
    v = tf.get_variable('v', initializer = tf.zeros_initializer(shape = [1]))
#在计算图g1中读取变量v的取值
with tf.Session(graph = g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope('', reuse = True):
        print(sess.run(tf.get_variable('v')))


g2 = tf.Graph()

with g2.as_default():
    v = tf.get_variable('v', initializer = tf.ones_initializer(shape = [1]))

with tf.Session(graph = g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope('', reuse = True):
        print(sess.run(tf.get_variable('v')))

