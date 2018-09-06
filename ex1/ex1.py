import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def pause():
    program_pause = input('Program paused. Press ENTER to continue.\n')

def warm_up_exercise():
    eye = tf.eye(5)
    return eye.eval()

def plot_data(X, y):
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(X, y, marker='x', c='r', s=10)
    ax.set_ylabel('Profit in $10,000s')
    ax.set_xlabel('Population of City in 10,000s')
    plt.show(block=False)
    
def compute_cost(X, y, theta):
    if isinstance(y, tf.Tensor):
        m = tf.constant(float(y.shape.as_list()[0]))
    else:
        m = tf.constant(float(len(y)))
    # conversion to tensorflow Tensors
    if isinstance(X, np.ndarray):
        X = tf.convert_to_tensor(X, tf.float32)
    if isinstance(y, np.ndarray):
        y = tf.convert_to_tensor(y, tf.float32)
    return (tf.constant(1.0) / (tf.constant(2.0) * m)
           ) * tf.reduce_sum(tf.square(tf.reshape(tf.matmul(X, theta), [-1]) - y))

def gradient_descent(X, y, theta, alpha, num_iters):
    if isinstance(y, tf.Tensor):
        m = tf.constant(float(y.shape.as_list()[0]))
    else:
        m = tf.constant(float(len(y)))
    if isinstance(y, np.ndarray):
        y = tf.convert_to_tensor(y, tf.float32)
    if isinstance(X, np.ndarray):
        X = tf.convert_to_tensor(X, tf.float32)
    j_history = tf.Variable(np.zeros(int(num_iters.eval())), 1)
    sess.run(j_history.initializer)
    for i in range(int(num_iters.eval())):
        temp0 = theta[0] - (alpha / m) * tf.reduce_sum(
                                                tf.multiply(tf.reshape(tf.matmul(X, theta), [-1]) - y, X[:, 0]))
        temp1 = theta[1] - (alpha / m) * tf.reduce_sum(
                                                tf.multiply(tf.reshape(tf.matmul(X, theta), [-1]) - y, X[:, 1]))
        theta.load([temp0.eval(), temp1.eval()])
        # tf.assign(theta[0], temp0)
        # tf.assign(theta[1], temp0)
        
        sess.run(tf.assign(j_history[i], compute_cost(X, y, theta).eval()))
    return theta, j_history

def run_all():

    print('Running warp_up_exercise\n 5x5 Identity Matrix: \n{}'.format(warm_up_exercise()))

    print('Plotting Data ...\n')
    data = pd.read_table('ex1data1.txt', sep=',', header=None)
    X, y = data[0].values, data[1].values
    plot_data(X, y)
    pause()

    X = pd.DataFrame([np.ones(len(X)), X]).T.values
    theta = tf.Variable(initial_value=[[0.0], [0.0]], dtype=tf.float32, name='theta')
    sess.run(theta.initializer)
    iterations = tf.constant(1500.0, dtype=tf.float32)
    alpha = tf.constant(0.03, dtype=tf.float32)

    print('\nTesting the cost function...\n')
    J = tf.Variable(initial_value=0.0, name='cost', dtype=tf.float32)
    sess.run(J.initializer)
    J.load(compute_cost(X, y, theta).eval())

    print('With theta = [0 ; 0]\nCost computed = {0:.6f}\n'.format(J.eval()))
    print('Expected cost value (approx) 32.07\n')

    theta.load([[-1.0], [2.0]])
    J.load(compute_cost(X, y, theta).eval())
                    
    print('\nWith theta = [-1, 2]\nCost computed = {0:.6f}\n'.format(J.eval()))
    print('Expected cost value (approx) 54.24\n')

    pause()
    optim = tf.train.GradientDescentOptimizer(0.01)
    print('\nRunning Gradient Descent ...\n')
    # theta, j_history = gradient_descent(X, y, theta, alpha, iterations)
    optim.compute_gradients(loss=compute_cost, var_list=theta)
    print('Theta found by gradient descent:\n')
    print('{0:.6f}'.format(theta.eval()))
    print('Expected theta values (approx)\n')
    print(' -3.6303\n 1.1664\n\n')


if __name__ == '__main__':
    run_all()