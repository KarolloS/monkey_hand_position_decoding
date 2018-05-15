import numpy as np
import time
import random as rand
import scipy.io as spio
import tensorflow as tf

# Import data from .mat file
data = spio.loadmat('data.mat', squeeze_me=True)
data_x = np.transpose(data['x'])
data_t = np.transpose(data['t'])
data_xt = np.transpose(data['xt'])
data_tt = np.transpose(data['tt'])

input_len = len(data_x[0,:])
output_len = len(data_t[0,:])
data_len = len(data_x[:,1])
test_len = len(data_xt[:,1])

# Set parameters
training_iteration = 40
batch_size = 100
total_batch = round(data_len/batch_size)

# TF graph input
x = tf.placeholder("float", [None, input_len], name='input')
y = tf.placeholder("float", [None, output_len], name='output')

# Create a model

# Set model weights
layers = [500]
W1 = tf.Variable(tf.random_normal([input_len, layers[0]], stddev=0.05), name='weight_1')
W2 = tf.Variable(tf.random_normal([layers[0], output_len], stddev=0.05), name='weight_2')
b = tf.Variable(tf.random_normal([output_len], stddev=0.1), name='bias')

with tf.variable_scope("W2W1x_b"):
    # Construct a linear model
    h = tf.nn.relu(tf.matmul(x, W1))
    model = tf.nn.softmax(tf.matmul(h, W2) + b) # Softmax

# Add summary ops to collect data
w1_h = tf.summary.histogram(W1.op.name, W1)
w2_h = tf.summary.histogram(W2.op.name, W2)
b_h = tf.summary.histogram(b.op.name, b)

# Create loss function
with tf.variable_scope("cost_function"):
    # Minimize error
    # cost_function = -tf.reduce_sum(y*tf.log(model)) # Cross entropy
    cost_function = tf.reduce_sum(tf.pow(y - model, 2)) # MSE
    # Create a summary to monitor the cost function
    tf.summary.scalar(cost_function.op.name, cost_function)

with tf.variable_scope("train"):
    # Optimizer
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost_function) # Gradient Descent
    optimizer = tf.train.AdadeltaOptimizer(1.0).minimize(cost_function) # AdaDelta

# Initializing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess: # don't use GPU
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess: # use GPU
    sess.run(init)

    # Set the logs writer
    summary_writer = tf.summary.FileWriter('BMI_log', sess.graph)

    print("Training started...")
    start_time = time.time()

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
            # create batch with random datapoints
            # each batch contains #batch_size datapoints selected randomly from the whole dataset
            batch_x = np.zeros((batch_size, input_len))
            batch_t = np.zeros((batch_size, output_len))
            for j in range(batch_size):
                k = round(rand.uniform(1, data_len-1))
                batch_x[j, :] = data_x[k,:]
                batch_t[j, :] = data_t[k, :]
            # Fit training
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_t})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_x, y: batch_t})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_x, y: batch_t})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    elapsed_time = time.time() - start_time
    print("Training completed! (" +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + ")")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: data_xt, y: data_tt}))

    # Get predictions for whole testing dataset
    pred = sess.run(model, feed_dict={x: data_xt})
    for i in range(test_len):
        a = np.argmax(pred[i,:])
        pred[i,:] = 0
        pred[i,a] = 1

    # Export predictions to .mat file
    spio.savemat('results.mat', {'pred':np.transpose(pred)})


#  tensorboard --logdir=D:\Studia\Inne\TensorFlow_Demo
