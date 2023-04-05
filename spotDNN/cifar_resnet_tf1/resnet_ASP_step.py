#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow as tf
import resnet_model


import cifar10
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/test/datasets/cifar-100-binary',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('train_dir', '/test/cifar_resnet_tf1/model_resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('ps_hosts', "localhost:5555", 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', "localhost:5557",'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('job_name', None, 'job name: worker or ps')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_boolean('issync', False, 'Whether synchronization')
tf.app.flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
tf.app.flags.DEFINE_string('dataset', "cifar100", """The dataset to use.""")

tf.app.flags.DEFINE_string('TF_FORCE_GPU_ALLOW_GROWTH', 'false', """""")
tf.app.flags.DEFINE_integer('max_steps', 50000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('resnet_size', 50, """The size of the ResNet model to use.""")
tf.app.flags.DEFINE_float('lr_adjust', 1, """The adjusted learning rate""")
tf.app.flags.DEFINE_integer('trytag', 0, """the try times tag""")
tf.app.flags.DEFINE_float('sleep', 0, """adjust gpu usage""")
tf.app.flags.DEFINE_boolean('prof', False, """Whether to profile.""")
# cifar10_resnet_v2_generator(resnet 14 32 50 110 152 200)
# resnet_v2(resnet 18 34 50 101 152 200)



tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.1      # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3

if FLAGS.dataset == "cifar10":
    _NUM_CLASSES = 10
elif FLAGS.dataset == "cifar100":
    _NUM_CLASSES = 100

_WEIGHT_DECAY = 2e-4


def train():
    enter_time = time.time()
    worker_hosts = FLAGS.worker_hosts.split(',')
    ps_hosts = FLAGS.ps_hosts.split(',')
    issync = FLAGS.issync
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == "worker":
        time.sleep(5)
        is_chief = (FLAGS.task_index == 0)
        
        if not(tf.gfile.Exists(FLAGS.train_dir)):
            tf.gfile.MakeDirs(FLAGS.train_dir)
            tf.gfile.MakeDirs(FLAGS.train_dir + '/timeline')

        file = FLAGS.train_dir + "/" + FLAGS.job_name + str(FLAGS.task_index) + \
               "_resnet" + str(FLAGS.resnet_size) + \
               "_b" + str(FLAGS.batch_size) + "_s" + str(FLAGS.max_steps) + "_" + str(FLAGS.trytag) + ".txt"
        loss_file = open(file, "w")
        loss_file.write("datetime\tg_step\tg_img\tloss_value\texamples_per_sec\tsec_per_batch\n")

        worker_device = "/job:worker/task:%d" % FLAGS.task_index
        if FLAGS.num_gpus > 0:
            gpu = (FLAGS.task_index % FLAGS.num_gpus)
            worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)            
     
        with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster
            )):

            global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)
            global_img = tf.get_variable(
                    'global_img', [],
                    initializer=tf.constant_initializer(0), trainable=False)
            img_op = tf.add(global_img, FLAGS.batch_size)
            img_update = tf.assign(global_img, img_op)

            decay_steps = 50000*350.0/FLAGS.batch_size
            batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            inputs, labels = cifar10.distorted_inputs(FLAGS.data_dir, FLAGS.dataset, FLAGS.batch_size)
            network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
            if FLAGS.dataset == "cifar10":
                labels = tf.one_hot(labels, 10, 1, 0)
            elif FLAGS.dataset == "cifar100":
                labels = tf.one_hot(labels, 100, 1, 0)
            logits = network(inputs, True)
            cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

            loss = cross_entropy + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr * FLAGS.lr_adjust)

            # Track the moving averages of all trainable variables.
            exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = exp_moving_averager.apply(tf.trainable_variables())

            # added by faye
            grads0 = opt.compute_gradients(loss) 
            grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]

            time.sleep( FLAGS.sleep )

            if issync:
                opt = tf.train.SyncReplicasOptimizer(
                    opt,
                    replicas_to_aggregate=len(worker_hosts),
                    total_num_replicas=len(worker_hosts),
                    variable_averages=exp_moving_averager,
                    variables_to_average=variables_to_average)
                if is_chief:
                    chief_queue_runners = opt.get_chief_queue_runner()
                    init_tokens_op = opt.get_init_tokens_op()

            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
                
            train_op = tf.group(apply_gradient_op, variables_averages_op)
            
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
                                     init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()),
                                     global_step=global_step,
                                     recovery_wait_secs=1)
                                     #save_model_secs=60)

            sess_config = tf.ConfigProto(
                allow_soft_placement=True, 
                log_device_placement=FLAGS.log_device_placement)

            if is_chief:
                print("Worker %d: Initializing session..." % FLAGS.task_index)
            else:
                print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
            
            # Get a session.
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

            print("Worker %d: Session initialization complete." % FLAGS.task_index)

            # Start the queue runners.
            if is_chief and issync:
                sess.run(init_tokens_op)
                sv.start_queue_runners(sess, [chief_queue_runners])
            #else:
            #    sv.start_queue_runners(sess=sess)

            """Train CIFAR-100 for a number of steps."""

            step = 0
            g_step = 0
            train_begin = time.time()
            InitialTime = train_begin - enter_time
            print("Batch size: @", FLAGS.batch_size)
            print("Initial time is @ %f" % InitialTime)
            print("Training begins @ %f" % train_begin)
            tag = 1
            batch_size_num = FLAGS.batch_size
            while g_step < FLAGS.max_steps:
                start_time = time.time()
                if FLAGS.prof:
                    profiler = tf.profiler.Profiler(graph=sess.graph)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss_value, g_step, g_img = sess.run([train_op, loss, global_step, img_update], \
                                                        feed_dict={batch_size: batch_size_num}, \
                                                        options=run_options, run_metadata=run_metadata)
                    profiler.add_step(step=step, run_meta=run_metadata)
                else:
                     _, loss_value, g_step, g_img = sess.run([train_op, loss, global_step, img_update], \
                                                        feed_dict={batch_size: batch_size_num})


                if tag:
                    fisrt_sessrun_done = time.time()
                    print("First sessrun time is @ %f" % (fisrt_sessrun_done - train_begin))
                    tag = 0
                    FirstSessRunTime = fisrt_sessrun_done - train_begin

                if step % 5 == 0:
                        duration = time.time() - start_time
                        examples_per_sec = batch_size_num / duration
                        sec_per_batch = float(duration)
                        format_str = ('[worker %d] local_step %d (global_steps %d, img_update %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (FLAGS.task_index, step, g_step, g_img, loss_value, examples_per_sec, sec_per_batch))
                        loss_file.write("%s\t%d\t%s\t%s\t%s\t%s\n" %(datetime.now(), g_step, g_img, loss_value, examples_per_sec, sec_per_batch))
                step += 1

            if FLAGS.prof:
                opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
                param_stats = profiler.profile_name_scope(options=opts)
                print('total parameters:', param_stats.total_parameters)
                opts = tf.profiler.ProfileOptionBuilder.float_operation()
                float_stats = profiler.profile_operations(opts)
                print('total Flops:', float_stats.total_float_ops)


            train_end = time.time()
            loss_file.write("TrainTime\t%f\n" %(train_end-train_begin))
            loss_file.write("InitialTime\t%f\n" %InitialTime)
            loss_file.write("FirstSessRunTime\t%f\n" %FirstSessRunTime)
            loss_file.close()
            
            # end of while
            sv.stop()
            # end of with

def main(argv=None):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = FLAGS.TF_FORCE_GPU_ALLOW_GROWTH
    train()

if __name__ == '__main__':
    tf.app.run()
