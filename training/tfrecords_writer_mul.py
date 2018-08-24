#!/usr/bin/python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import random

import numpy as np
import tensorflow as tf
from utils.input_trans_seen import input_interface

from utils.input_pre_trans import pre_trans_no_pot


class MyProcess(multiprocessing.Process):
    def __init__(self, process_id, name, lock, test=False):
        multiprocessing.Process.__init__(self)
        self.process_id = process_id
        self.name = name
        self.test = test
        self.lock = lock

    def run(self):
        print("开始进程：" + self.name)
        write_records(self.process_id, self.lock, self.test)
        print("退出进程：" + self.name)


def build_bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def build_int64_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def build_float_feature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[v]))


def build_data(sample, label, legal, score):
    data = tf.train.Example(features=tf.train.Features(feature={
        'sample': build_bytes_feature(sample),
        'label': build_int64_feature(label),
        'legal': build_bytes_feature(legal),
        'score': build_float_feature(score)
    }))
    return data


def write_records(cnt, lock, test=False):
    LINE_NUM = 100000
    file_src = 'C:/Users/humy/Desktop/牌谱/200W真人/ddz%d.txt'
    train_save_path = 'E:/cnn_data/training'
    lock.acquire()
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    lock.release()
    test_save_path = 'E:/cnn_data/test'
    if test:
        lock.acquire()
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        lock.release()
    train_file_pattern = train_save_path + '/ddz_training_data_%.4d.tfrecords'
    test_file_pattern = test_save_path + '/ddz_test_data_%.4d.tfrecords'
    train_file_no = cnt * 1000 + 1
    test_file_no = cnt * 1000 + 1
    train_line_cnt = 0
    test_line_cnt = 0

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    with tf.Session(config=config):
        with open(file_src % cnt) as fp:
            train_writer = tf.python_io.TFRecordWriter(train_file_pattern % train_file_no)
            if test:
                test_writer = tf.python_io.TFRecordWriter(test_file_pattern % test_file_no)
            for line in fp.readlines():
                line = line.strip()
                game_str, role = pre_trans_no_pot(line)
                # print(game_str, role)
                sample, label, legal, score = input_interface(game_str, role)
                if sample is not None and len(label) > 0:
                    rd = random.randint(0, len(label) - 1)
                    for i in range(len(label)):
                        if test and i == rd and len(label) > 1:
                            test_writer.write(build_data(sample=np.array(sample[i], dtype=np.uint8).tobytes(),
                                                         label=label[i],
                                                         legal=np.array(legal[i], dtype=np.uint8).tobytes(),
                                                         score=score).SerializeToString())
                            test_line_cnt += 1
                            if test_line_cnt >= LINE_NUM:  # 文件结束条件
                                test_writer.close()
                                test_line_cnt = 0
                                test_file_no += 1
                                test_writer = tf.python_io.TFRecordWriter(test_file_pattern % test_file_no)
                        else:
                            train_writer.write(build_data(sample=np.array(sample[i], dtype=np.uint8).tobytes(),
                                                          label=label[i],
                                                          legal=np.array(legal[i], dtype=np.uint8).tobytes(),
                                                          score=score).SerializeToString())
                            train_line_cnt += 1
                            if train_line_cnt >= LINE_NUM:  # 文件结束条件
                                train_writer.close()
                                train_line_cnt = 0
                                train_file_no += 1
                                train_writer = tf.python_io.TFRecordWriter(train_file_pattern % train_file_no)
                else:
                    print('from process:' + str(cnt))
                    print(line)
                    print(game_str, role)
            train_writer.close()
            if test:
                test_writer.close()


if __name__ == '__main__':
    # 创建新线程
    process_list = []
    lock = multiprocessing.Lock()
    for p in range(5):
        t = MyProcess(p, "Process-" + str(p), lock)
        t.start()
        process_list.append(t)

    for process in process_list:
        process.join()
