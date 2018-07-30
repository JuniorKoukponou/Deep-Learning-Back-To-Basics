
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    import numpy as np
    import tensorflow as tf
        
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import axes3d


def CreateOneLayer(X, n_neurones):
    W = tf.get_variable("W", [X.get_shape()[1].value, n_neurones], tf.float32)
    b = tf.get_variable("b", [n_neurones], tf.float32)
    return tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))

def model():
    if 'x' not in _model_ :
        with tf.variable_scope(_model_+"layer1", reuse=tf.AUTO_REUSE):
            model = CreateOneLayer(X, 1)
    else :
        with tf.variable_scope(_model_+"layer1", reuse=tf.AUTO_REUSE):
            z0 = CreateOneLayer(X, 2)
        with tf.variable_scope(_model_+"layer2", reuse=tf.AUTO_REUSE):
            model = CreateOneLayer(z0, 1)
    return model

def CreateTrainingOperator(mod, learning_rate, labels):
    with tf.variable_scope(_model_):
        loss_op = tf.reduce_mean(tf.square(tf.subtract(mod, labels)))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
        tf.summary.scalar("loss", loss_op)

    return train_op, loss_op

def predict_single(x1, x2, _model_) :
    z0 = model()
    saver = tf.train.Saver()
    saver.restore(sess, "./models/"+_model_+".ckpt")
    X_in = np.array([x1, x2]).reshape([-1, 2])
    pred = sess.run(z0, feed_dict={X: X_in})
    and_single = np.int(np.round(np.reshape(pred, 1)[0]))
    return and_single

if __name__ == '__main__' :
    import sys
    _model_ = sys.argv[3]

    tf.reset_default_graph()
    g = tf.Graph()
    sess = tf.Session(graph = g)

    with g.as_default():

        X = tf.placeholder(tf.float32, [None, 2], name="X-input")
        y = tf.placeholder(tf.float32, [None, 1], name="y-output")

        mod = model()
        
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
                            [0]])
                                
        
        print(predict_single(x1=sys.argv[1], x2=sys.argv[2],  _model_=sys.argv[3]))

# python3 predict.py 1 0 and