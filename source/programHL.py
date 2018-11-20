import tensorflow as tf
import numpy as np
import numpy.random as rd
from nn18_ex2_load import load_isolet
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import transpose



Xa,C,X_test,C_test=load_isolet();
print(Xa.shape);
print(C.shape);
print(X_test.shape);
print(C_test.shape);

architecture=0;
#We create the C arrays with this function
def createC(array):
    res=np.zeros((len(array),26));
    
    for i in range(len(array)):
        res[i][array[i]-1]=1;
    return res;
#normalize function

def normalizeDataNN(array):
    m=transpose(array);
    res=np.zeros((len(array[0]),len(array)),dtype=float);
    for i in range (len(m)):
        maximum=max(m[i]);
        minimum=min(m[i]);
        for j in range(len(m[i])):
            res[i][j]=float(m[i][j]-minimum)/float(maximum-minimum);
    return transpose(res);

Ca=createC(C);
Ca_Test=createC(C_test);
Xa=normalizeDataNN(Xa);
Ca=normalizeDataNN(Ca);
X_test=normalizeDataNN(X_test);
Ca_Test=normalizeDataNN(Ca_Test);

# Give the dimension of the data and chose the number of hidden layer
if architecture==0:
    n_in = 300
    n_out = 26
    n_hidden = 20
    
    # Set the variables
    W_hid = tf.Variable(rd.randn(n_in,n_hidden) / np.sqrt(n_in),trainable=True)
    b_hid = tf.Variable(np.zeros(n_hidden),trainable=True)
    # Lets try adding another hiddel layer
    W_hid2 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
    b_hid2 = tf.Variable(np.zeros(n_hidden),trainable=True)
    #######################################
    w_out = tf.Variable(rd.randn(n_hidden,n_out) / np.sqrt(n_in),trainable=True)
    b_out = tf.Variable(np.zeros(n_out))
    
    # Define the neuron operations
    x = tf.placeholder(shape=(None,300),dtype=tf.float64)
    y = tf.nn.tanh(tf.matmul(tf.matmul(x,W_hid) + b_hid,W_hid2+b_hid2));
    z = tf.nn.softmax(tf.matmul(y,w_out) + b_out)
    
    
    
    z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))
    
    
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    init = tf.global_variables_initializer() # Create an op that will
    sess = tf.Session()
    sess.run(init) # Set the value of the variables to their initialization value
    
    
    
    # Re init variables to start from scratch
    sess.run(init)
    
    # Create some list to monitor how error decreases
    test_loss_list = []
    train_loss_list = []
    
    test_acc_list = []
    train_acc_list = []
    
    # Create minibtaches to train faster
    k_batch = 40
    X_batch_list = np.array_split(Xa,k_batch)
    labels_batch_list = np.array_split(Ca,k_batch)
    initial=np.random.randint(0,5500-100);
    validation_set=Xa[initial:initial+100][:];
    Ca_validation_set=Ca[initial:initial+100][:]
    maxacc=0.0;
    saver = tf.train.Saver()
    epochs=0;
    #Training with validation set
    for k in range(250):
        
        # Run gradient steps over each minibatch
        for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
            sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})
        
        
        # Compute the errors over the whole dataset
        train_loss = sess.run(cross_entropy, feed_dict={x:Xa, z_:Ca})
        test_loss = sess.run(cross_entropy, feed_dict={x:validation_set,z_:Ca_validation_set});
        
        # Compute the acc over the whole dataset
        train_acc = sess.run(accuracy, feed_dict={x:Xa, z_:Ca})
        test_acc = sess.run(accuracy, feed_dict={x:validation_set,z_:Ca_validation_set});
        if maxacc<test_acc:
            save_path = saver.save(sess, "/tmp/model.ckpt")
            maxacc=test_acc;
            epochs=k;
            #Save state 
        # Put it into the lists
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        
        
        if np.mod(k,10) == 0:
            print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))
    print('iteration {} test accuracy: {:.3f}'.format(k+1,maxacc));
    fig,ax_list = plt.subplots(1,2)
    ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)
    ax_list[0].plot(test_loss_list, color='green', label='testing', lw=2)
    ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
    ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)
    
    ax_list[0].set_xlabel('training iterations')
    ax_list[1].set_xlabel('training iterations')
    ax_list[0].set_ylabel('Cross-entropy')
    ax_list[1].set_ylabel('Accuracy')
    plt.legend(loc=2)
    plt.show();
    #Now we use the test set
    test_loss_list2 = []
    train_loss_list2 = []
    
    test_acc_list2 = []
    train_acc_list2 = []
    saver.restore(sess, "/tmp/model.ckpt")
    for k in range(epochs):
        
        # Run gradient steps over each minibatch
        for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
            sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})
        
        
        # Compute the errors over the whole dataset
        train_loss = sess.run(cross_entropy, feed_dict={x:Xa, z_:Ca})
        test_loss = sess.run(cross_entropy, feed_dict={x:validation_set,z_:Ca_validation_set});
        
        # Compute the acc over the whole dataset
        train_acc = sess.run(accuracy, feed_dict={x:Xa, z_:Ca})
        test_acc = sess.run(accuracy, feed_dict={x:X_test,z_:Ca_Test});
        if maxacc<test_acc:
            maxacc=test_acc;
            #Save state 
        # Put it into the lists
        test_loss_list2.append(test_loss)
        train_loss_list2.append(train_loss)
        test_acc_list2.append(test_acc)
        train_acc_list2.append(train_acc)
        
        
        if np.mod(k,10) == 0:
            print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))
    fig,ax_list = plt.subplots(1,2)
    ax_list[0].plot(train_loss_list2, color='blue', label='training', lw=2)
    ax_list[0].plot(test_loss_list2, color='green', label='testing', lw=2)
    ax_list[1].plot(train_acc_list2, color='blue', label='training', lw=2)
    ax_list[1].plot(test_acc_list2, color='green', label='testing', lw=2)
    
    ax_list[0].set_xlabel('training iterations')
    ax_list[1].set_xlabel('training iterations')
    ax_list[0].set_ylabel('Cross-entropy')
    ax_list[1].set_ylabel('Accuracy')
    plt.legend(loc=2)
    print('iteration {} test accuracy: {:.3f}'.format(k+1,maxacc));
    plt.show();