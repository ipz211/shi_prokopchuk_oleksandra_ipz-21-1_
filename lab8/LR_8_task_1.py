import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Parameters
n_samples, batch_size, num_steps = 1000, 100, 20000
learning_rate = 0.0001

# Generate synthetic data
X_data = np.random.uniform(0, 1, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, np.sqrt(2), (n_samples, 1))

# Placeholders for input data
X = tf.placeholder(tf.float32, shape=(batch_size, 1), name='X')
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y')

# Define the model
with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1), mean=0.0, stddev=0.1), name='slope')
    b = tf.Variable(tf.zeros((1,)), name='bias')

# Predictions and loss
y_pred = tf.matmul(X, k) + b
loss = tf.reduce_sum((y - y_pred) ** 2)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Training
display_step = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        # Sample a batch of data
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        # Perform optimization and calculate loss
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b], feed_dict={X: X_batch, y: y_batch})

        # Display progress
        if (i + 1) % display_step == 0:
            print('Epoch %d: Loss=%.8f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_val[0][0], b_val[0]))