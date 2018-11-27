## Hello TensorFlow

import matplotlib, numpy, pandas, scipy, sklearn
import pandas as pd
import matplotlib as plt
import numpy as np
import tensorflow as tf

import os
os.chdir("./MG/tf_data")

# 노드 생성
hello = tf.constant("Hello, Tensor!")

# 세션 생성
sess = tf.Session()

# 세션.실행
print(sess.run(hello))         # b : Bytes 스트링

## Computational Graph

node1 = tf.constant(3.0, tf.float32)                  # 그래프 빌드
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 바로 실행하면 원하는 값 출력X
print(node1 , node2)
print(node3)

sess = tf.Session()                                   # 세션 실행
print(sess.run([node1, node2]))
print(sess.run(node3))

# 플레이스홀드 (노드) 생성
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

# 피드_딕트로 값 지정
sess.run(adder_node, feed_dict={a: 3, b: 4.5})
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

## 타입. 랭크, 쉐입


# 1. Linear Regression의 Hypothesis와 cost
## Regression
## (LInear) Hypothesis : H(x) = Wx + b
## Cost function : (H(x) - y)^2 / m
## Goal : Minimize cost(W, b)  비용함수의 최소화

# 그래프 필드
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')              # trainable 한 Variable
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))              # reduce_mean : 평균

# 매직구현
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 세션 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())                          # W, b 변수 후 반드시 실행할 것

## Fit the Line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

## 플레이스 홀드를 이용한 선형회귀
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)










