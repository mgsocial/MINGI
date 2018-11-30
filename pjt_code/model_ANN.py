# 함수불러오기
import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 셋 x, y의 데이터 값
x_data = X_train_scaled
y_data = y_train.reshape(1281166,1)

# 데이터를 담는 플레이스 홀더 정의
X = tf.placeholder(tf.float64, shape=[None, 18])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기 정의
W = tf.Variable(tf.random_uniform([18,1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# 다중회귀 방정식 정의
y = tf.matmul(X, W) + b

# 비용함수 생성
rmse = tf.sqrt(tf.reduce_mean(tf.square( y - Y)))

a = 0.1

# 최소화 함수
gradient_decent = tf.train.GradientDescentOptimizer(a).minimize(rmse)

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        W_, b_, rmse_, _ = sess.run([W, b, rmse, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i + 1) % 300 == 0:
            print(i + 1, W_[0], W_[1], W_[2], W_[3], W_[4], W_[5], W_[6], W_[7], W_[8], W_[9], W_[10],
            W_[11], W_[12], W_[13], W_[14], W_[15], W_[16], W_[17], b_, rmse_)
            new_y = sess.run(y, feed_dict={X: X_test_scaled})