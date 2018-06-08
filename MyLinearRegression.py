import DataHelper as dh
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_DIRECTORY = 'tf_model'
MODEL_NAME = 'model'

train_data = dh.get_train_data()
test_data = dh.get_test_data()

x_train = train_data[0].values
y_train = train_data[1].values

x_test = test_data[0].values
y_test = test_data[1].values

iterations = 2001
learning_rate = 0.01
display_step = 50

x = tf.placeholder(dtype=tf.float32, name='input')
y = tf.placeholder(dtype=tf.float32)


def linear_regression():
    w = tf.Variable(3.0, tf.float32, name='weight')
    b = tf.Variable(1.0, tf.float32, name='bias')
    y_pred = tf.add(tf.multiply(w, x), b, name='prediction')

    loss_function = tf.reduce_mean(tf.square(y_pred - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss_function)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tf.train.write_graph(sess.graph_def, MODEL_DIRECTORY, MODEL_NAME + '.pbtxt')
        for i in range(0, iterations):
            sess.run(optimizer, feed_dict={x: x_train, y: y_train})
            if i % display_step == 0:
                print(i, sess.run(w), sess.run(b))

        training_cost = sess.run(loss_function, feed_dict={x: x_train, y: y_train})
        print()
        print("Training cost=", training_cost, "w=", sess.run(w), "b=", sess.run(b))
        print("Testing... (Mean square loss_function Comparison)")

        testing_cost = sess.run(loss_function, feed_dict={x: x_test, y: y_test})
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss_function difference:", abs(training_cost - testing_cost))
        saver.save(sess, MODEL_DIRECTORY + '/' + MODEL_NAME + '.ckpt')


def save_model():
    input_graph_path = MODEL_DIRECTORY + '/' + MODEL_NAME + '.pbtxt'
    checkpoint_path = MODEL_DIRECTORY + '/' + MODEL_NAME + '.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "prediction"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = MODEL_DIRECTORY + '/frozen.pb'
    output_optimized_graph_name = MODEL_DIRECTORY + '/optimized.pb'
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input"],  # an array of the input node(s)
        ["prediction"],  # an array of output nodes
        tf.float32.as_datatype_enum)

    # Save the optimized graph

    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())


linear_regression()
save_model()