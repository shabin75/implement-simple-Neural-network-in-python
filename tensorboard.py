import tensorflow as tf

inputs = tf.placeholder('float', shape=[None, 2], name="input")
target = tf.placeholder('float', shape=[None, 1], name="input")
# cost=[]
inputbBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4), dtype='float', name="input_bias")

input_weights = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.4), dtype='float',name="hidden_weights")
#tf.summary.histogram(name="waights__1",values=input_weights)

hidden_bias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4), dtype='float', name="hidden_bias")

outputWeights = tf.Variable(initial_value=tf.random_normal(shape=[3, 1], stddev=0.4), dtype='float',
                            name="output_weights")
#tf.summary.histogram(name="waights__2",values=outputWeights)

hiddenlayer = tf.matmul(inputs, input_weights) + inputbBias
hiddenlayer = tf.sigmoid(hiddenlayer, name="hidden_layer_activation")

output = tf.matmul(hiddenlayer, outputWeights) + hidden_bias
output = tf.sigmoid(output, name="output_layer_activation")
tf.print(output)
cost = tf.squared_difference(target, output)
cost = tf.reduce_mean(cost)
tf.summary.scalar("error", cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)

inp = [[1, 1], [1, 0], [0, 1], [0, 0]]
out = [[0], [1], [1], [0]]
itera = 5000

import datetime
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    merge_summary=tf.summary.merge_all()
    file_name="./suumary_log/run"+datetime.datetime.now().strftime("%M-%m-%d--%H-m-s")
    writer=tf.summary.FileWriter(file_name,sess.graph)
    for i in range(itera):
        err,v,summary_out= sess.run([cost, optimizer,merge_summary], feed_dict={inputs: inp, target: out})
        #print(i,err)
        writer.add_summary(summary_out,i)

    while True:
        Inpu=[[0,0]]
        Inpu[0][0]=input("enter the first input:")
        Inpu[0][1]=input("enter the second input:")
        print(sess.run([output],feed_dict={inputs:Inpu})[0][0])
