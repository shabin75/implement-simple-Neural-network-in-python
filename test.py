import tensorflow as tf

input1 = tf.placeholder(tf.float32, shape=[None, 2], name="input")
target = tf.placeholder(tf.float32, shape=[None, 1], name="target")

hiddenweight = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.4), dtype='float',
                          name="hidden_weights")

tf.summary.histogram(name="waights__2",values=hiddenweight)

hiddenbias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4), dtype='float', name="hidden_bias")
hiddenout = tf.add(tf.matmul(input1, hiddenweight), hiddenbias)
hiddenpred = tf.sigmoid(hiddenout)

outerweight = tf.Variable(initial_value=tf.random_normal(shape=[3, 1], stddev=0.4), dtype='float', name="outer_weights")
outerbias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4), dtype='float', name="outer_bias")
out = tf.add(tf.matmul(hiddenpred, outerweight), outerbias)
outerpred = tf.sigmoid(out)

cost = tf.squared_difference(outerpred, target)
recost = tf.reduce_mean(cost)
opt = tf.train.AdamOptimizer().minimize(recost)
epoch = 4000

file1 = "./python_workspace/humandetection"

with tf.Session() as sess:
    r = tf.global_variables_initializer().run()
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(file1, sess.graph)
    #sess.run(r)
    for i in range(epoch):
        o,_,outp = sess.run([recost, opt,merge_summary],feed_dict={input1: [[0, 0], [0, 1], [1, 0], [1, 1]], target: [[0], [1], [1], [0]]})



        writer.add_summary(outp)
        # print(o)
        # print(hh)

    # while True:
    #     row = int(input("Enter number of rows in the matrix: "))
    #
    #     column = int(input("Enter number of columns in the matrix: "))
    #     inputmatrix = []
    #     print("Enter the %s x %s matrix: " % (row, column))
    #     for i in range(row):
    #         inputmatrix.append(list(map(int, input().rstrip().split())))
    #     outfinal = sess.run(outerpred, feed_dict={input1: inputmatrix})
    #     # print(outfinal)
    #
    #     if outfinal < 0.5:
    #         print("0")
    #     else:
    #         print("1")