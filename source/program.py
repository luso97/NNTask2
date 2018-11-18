import tensorflow as tf
import numpy as np
import numpy.random as rd
from nn18_ex2_load import load_isolet
import matplotlib.pyplot as plt




Xa,C,X_test,C_test=load_isolet();
print(Xa.shape);
print(C.shape);
print(X_test.shape);
print(C_test.shape);

#We create the C arrays with this function
def createC(array):
    res=np.zeros((len(array),26));
    
    for i in range(len(array)):
        res[i][array[i]-1]=1;
    return res;
Ca=createC(C);
Ca_Test=createC(C_test);

#We create the variables
W = tf.Variable(rd.randn(300,26),trainable=True);
b = tf.Variable(np.zeros(26),trainable=True);

x = tf.placeholder(shape=(None,300),dtype=tf.float64);

y = tf.nn.softmax(tf.matmul(x,W) + b);


y_ = tf.placeholder(shape=(None,26),dtype=tf.float64)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # It only updates variable with it is run

#Session run



init = tf.global_variables_initializer() # Create an op that will initialize the variable on demand
sess = tf.Session()



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))



# Re init variables to start from scratch
sess.run(init)

# Create lists for monitoring
test_error_list = []
train_error_list = []
test_acc_list = []
train_acc_list = []
k_batch = 600
X_batch_list = np.array_split(Xa,k_batch)
labels_batch_list = np.array_split(Ca,k_batch)
for k in range(20):
    # Compute a gradient step
    for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, y_:labels_minibatch})
    
    
    # Compute the losses on training and testing sets for monitoring
    train_err = sess.run(cross_entropy, feed_dict={x:Xa,y_:Ca})
    test_err = sess.run(cross_entropy, feed_dict={x:X_test,y_:Ca_Test});
    train_acc = sess.run(accuracy, feed_dict={x:Xa, y_:Ca})
    test_acc = sess.run(accuracy, feed_dict={x:X_test, y_:Ca_Test})
    test_error_list.append(test_err);
    train_error_list.append(train_err);
    train_acc_list.append(train_acc);
    test_acc_list.append(test_acc);


fig,ax = plt.subplots(1)
ax.plot(train_error_list, color='blue', label='training', lw=2)
ax.plot(test_error_list, color='green', label='testing', lw=2)
ax.plot(train_acc_list, color='red', label='training', lw=2)
ax.plot(test_acc_list, color='brown', label='testing', lw=2)
ax.set_xlabel('Training epoch')
ax.set_ylabel('Cross-entropy')
plt.legend()
plt.show(); 

print("end");





