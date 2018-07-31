#!/usr/bin/env python3

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# -------->

def CreateOneLayer(X, n_neurones):
    W = tf.get_variable("W", [X.get_shape()[1].value, n_neurones], tf.float32)
    b = tf.get_variable("b", [n_neurones], tf.float32)
    return tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))

def model(_model_):
    if 'x' not in _model_ :
        with tf.variable_scope(_model_+"layer1", reuse = tf.AUTO_REUSE):
            model = CreateOneLayer(X, 1)
    else :
        print("\nDealing with a <X...> model ... \n")
        with tf.variable_scope(_model_+"layer1", reuse = tf.AUTO_REUSE):
            z0 = CreateOneLayer(X, 2)
        with tf.variable_scope(_model_+"layer2", reuse = tf.AUTO_REUSE):
            model = CreateOneLayer(z0, 1)
    return model

def CreateTrainingOperator(mod, learning_rate, labels):
    with tf.variable_scope(_model_):
        loss_op = tf.reduce_mean(tf.square(tf.subtract(mod, labels)))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
        tf.summary.scalar("loss", loss_op)
    return train_op, loss_op

def train(X_train, y_train, n_iterations, _model_) :
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summ_op = tf.summary.merge_all()
    sess.run(init_op)
    writer = tf.summary.FileWriter("./models/" + _model_ + "_log", g)
        
    for step in range(n_iterations + 1):
        _, cost = sess.run([training_op, loss_op], feed_dict={X: X_train, y: y_train})
        if step % 100 == 0:
            writer.add_summary(sess.run(summ_op, feed_dict={X: X_train, y: y_train}), step)
            print("Step : {: <5} - Lost = {: <23}".format(step, cost))
            
    save_path = saver.save(sess, "./models/"+_model_+".ckpt")
    sess.close()
    print("Model trained. Session saved in", save_path)

def predict(_model_) :
    result = model(_model_)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "./models/"+_model_+".ckpt")
    
    span = np.linspace(0, 1, 100)
    x1, x2 = np.meshgrid(span, span)
    X_in = np.column_stack([x1.flatten(), x2.flatten()])
    vals = np.reshape(sess.run(result, feed_dict={X: X_in}), x1.shape)

    return vals

def predict_single(x1, x2, _model_) :
    z0 = model(_model_)
    saver = tf.train.Saver()
    saver.restore(sess, "./models/"+_model_+".ckpt")
    X_in = np.array([x1, x2]).reshape([-1, 2])
    pred = sess.run(z0, feed_dict={X: X_in})
    single = np.int(np.round(np.reshape(pred, 1)[0]))
    return single


def visualise(span, vals) :
    fig = plt.figure(figsize = (20, 10))
    xv, yv = np.meshgrid(span, span)
    ax = fig.gca(projection='3d')
    cset = ax.contourf(xv, yv, vals, zdir='z', offset=span.min() - 0.3, cmap=cm.coolwarm)
    cset = ax.contourf(xv, yv, vals, zdir='x', offset=span.min() - 0.1, cmap=cm.coolwarm)
    cset = ax.contourf(xv, yv, vals, zdir='y', offset=span.max() + 0.1, cmap=cm.coolwarm)
    p = ax.plot_surface(xv, yv, vals, cmap=cm.coolwarm)

    ax.set_xlabel('x1')
    ax.set_xlim(span.min() - 0.1, span.max())
    ax.set_ylabel('y1')
    ax.set_ylim(span.min(), span.max() + 0.1)
    ax.set_zlabel('h')
    ax.set_zlim(span.min() - 0.3, span.max() + 0.1)
    
    cb = fig.colorbar(p, shrink = 0.5)
    plt.show()
    


if __name__ == '__main__' :
    
    _model_ = 'or'

    tf.reset_default_graph()
    g = tf.Graph()
    sess = tf.Session(graph = g)

    with g.as_default():

        X = tf.placeholder(tf.float32, [None, 2], name="X")
        y = tf.placeholder(tf.float32, [None, 1], name="y")

        mod = model(_model_)
        
        training_op, loss_op = CreateTrainingOperator(mod, 0.1, y) 
        
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
    
        writer = tf.summary.FileWriter("./models/"+_model_+"_log", g)
        loss_summ = tf.summary.scalar("loss", loss_op)
        
    
        X_train = np.array([[0, 0], 
                            [0, 1], 
                            [1, 0], 
                            [1, 1]])

        y_train = np.array([[0], 
                            [1],
                            [1], 
                            [1]])
                            
                                
        train(X_train, y_train, 500, _model_ = _model_)

        print("\nTraining done ... \n")

        vals = predict(_model_)

        print(" x1| x2| x1 {} x2".format(_model_.upper()))
        print("---+---+----------")
        print(" 0 | 0 | {:.3f}".format(vals[0][0]))
        print(" 0 | 1 | {:.3f}".format(vals[0][-1]))
        print(" 1 | 0 | {:.3f}".format(vals[-1][0]))
        print(" 1 | 1 | {:.3f}".format(vals[-1][-1]))
        
        span = np.linspace(0, 1, 100)
        visualise(span, vals)

        print(predict_single(1, 0, _model_ = _model_))