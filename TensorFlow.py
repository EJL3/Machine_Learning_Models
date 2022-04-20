                            # Tensorflow combined with scikitlearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn import datasets
from sklearn.preprocessing import normalize
from tensorflow.python.framework import ops

'''
# 2d array:-

with tf.compat.v1.Session() as sess:

    l = tf.constant([[1, 2, 3], [4, 5, 6]])

    d = np.array([[1, 2, 3], [4, 5, 6]])
    tf.constant(d)

    d = tf.compat.v1.linspace(start=2.0, stop=10.0, num=20)

    o = sess.run(d)
    print(o)
'''
# -------------------------------------------------------------------------------------------------------------X
'''
    #Variables:-

     init_vals = tf.compat.v1.random_normal((1,3),8,7)
     print(init_vals.shape)

     var = tf.Variable(init_vals)

     init = tf.compat.v1.global_variables_initializer()

     sess.run(init)
     print(sess.run(var))
'''
# -------------------------------------------------------------------------------------------------------------X
'''
     #Placeholders:-

     ph = tf.compat.v1.placeholder(tf.int16)
     d = tf.multiply(ph, 2)

     print(sess.run(d, feed_dict={ph: 5}))
     print(sess.run(d, feed_dict={ph: 15}))
'''
# --------------------------------------------------------------------------------------------------------------X

ops.reset_default_graph()
sess = tf.compat.v1.Session()

bc = datasets.load_breast_cancer()

print("\n")
print(bc.data.shape)
print("\n")
print(bc.data)
print("\n")
print(bc.target)
print("\n")
print(bc.feature_names)

# Splitting the data into training and testing sets

X_vals = bc.data
y_vals = bc.target

train_indices = np.random.choice(len(X_vals), round(len(X_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(X_vals))) - set(train_indices)))

print("\n")
print(test_indices)
print("\n")
print(train_indices)
print("\n")

x_vals_train = X_vals[train_indices]
x_vals_test = X_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Creating placeholders for input data and target data

batch_size = 25
tf.compat.v1.disable_eager_execution() # To disable immediate execution

x_data = tf.compat.v1.placeholder(shape=[None,30], dtype=tf.float32)
y_target = tf.compat.v1.placeholder(shape=[None,1], dtype=tf.float32)

A = tf.Variable(tf.compat.v1.random_normal(shape=[30,1], dtype=tf.float32))
B = tf.Variable(tf.compat.v1.random_normal(shape=[1,1]))

model_output = tf.add(tf.matmul(x_data, A), B)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

prediction = tf.round(tf.nn.sigmoid(model_output))
prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)

accuracy = tf.reduce_mean(prediction_correct)

# Train Model

loss_vec = []
train_acc =[]
test_acc = []

for i in range(1000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

    loss_vec.append(temp_loss)

    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})

    train_acc.append(temp_acc_train)

    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})

    test_acc.append(temp_acc_test)

    print(test_acc)
    print("\n")
    print(train_acc)

# Plot graph

plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

plt.plot(train_acc, 'k-', label='Train set accuracy')
plt.plot(test_acc, 'r--', label='Test set accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
