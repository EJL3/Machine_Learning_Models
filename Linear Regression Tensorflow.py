import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
tf.compat.v1.disable_eager_execution()

# Generating data

n_samples = 10000
x = np.linspace(0, 10, n_samples)
y = 2 * x + 5

# Randomly init model parameters

A = tf.Variable(tf.compat.v1.random_normal(shape=[1, 1]))
B = tf.Variable(tf.compat.v1.random_normal(shape=[1, 1]))

# Placeholders to take training data

x_data = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)

# Calculate model output

model_output = tf.add(tf.multiply(x_data, A), B)

# Calculate model loss

loss = tf.reduce_mean(tf.pow(model_output - y_target, 2)) / (n_samples * 2)

# Minimize loss by Gradient Descent method

gd = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5)
train_process = gd.minimize(loss)

loss_info = []
batch_size = 100

with tf.compat.v1.Session() as sess:

    sess.run(A.initializer)
    sess.run(B.initializer)

    for i in range(5001):

        rand_index = np.random.choice(len(x), size=batch_size)
        rand_x = x[rand_index]
        rand_y = y[rand_index]
        sess.run(train_process, feed_dict={x_data: np.transpose([rand_x]), y_target: np.transpose([rand_y])})
        loss_info.append(sess.run(loss, feed_dict={x_data: np.transpose([rand_x]), y_target: np.transpose([rand_y])}))

        if i % 1000 == 0:

            print("\n")

            print(sess.run([A, B]))

            tf.compat.v1.summary.FileWriter('tensorboard/log', sess.graph)
            print("\n")

            print("\nCoef after 5k iterations: ", sess.run([A, B]))

            print("\n")

            print("Loss after {} iterations: {}".format(i, loss_info[-1]))