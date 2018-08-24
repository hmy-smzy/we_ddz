import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.app.flags.DEFINE_integer('server', 4, 'number of server')

FLAGS = tf.app.flags.FLAGS


def server_start():
    worker_hosts = []
    port = 10935
    for i in range(FLAGS.server):
        worker = "localhost:%d" % (port + i)
        worker_hosts.append(worker)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    # config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.46
    config.gpu_options.allow_growth = True

    cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})

    servers = []
    for i in range(len(worker_hosts)):
        servers.append(tf.train.Server(cluster_spec, job_name="worker", task_index=i, config=config))
    for i in servers:
        i.join()


if __name__ == '__main__':
    server_start()
