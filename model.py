import tensorflow as tf
import numpy as np
import sys
import os
import heapq
import math

class DeepFM:
    def __init__(self):
        self.train = norm_train
        self.valid = norm_valid
        self.add_embedding_layer()
        self.add_placeholders()
        # initialize tuning options
        self.user_layer = [512, 64]
        self.item_layer = [1024, 64]
        self.lr = 0.0001
        self.max_epoch = 50
        self.batch_size = 256
        self.topK = 10
        # initialize layers
        self.add_embedding_layer()
        self.add_loss()
        self.add_train_step()
        self.check_point = args.check_point
        self.init_sess()


    def add_placeholder(self):
        self.user = tf.placeholder(tf.int32)
        self.item  = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)


    def add_embedding_matrix(self):
        self.user_embedding = tf.convert_to_tensor(item_user)
        self.item_embedding = tf.transpose(self.user_embedding)


    def add_embedding_layer(self):
        user_input = tf.nn.embedding_lookup(self.user_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("user_layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("item_layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)


    def add_loss(self):
        rate = self.rate
        losses =  rate * tf.log(self.y_) + (1 - rate) * tf.log(1 - self.y_)
        loss = -tf.reduce_sum(losses)
        self.loss = loss


    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)


    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if os.path.exists(self.check_point):
            [os.remove(f) for f in os.listdir(self.check_point)]
        else:
            os.mkdir(self.check_point)


   def run(self):
       # gain :  Normalized Discounted Cumulative Gain
        optimized= {'hit_ratio':-1, 'gain' : -1, 'epoch' : -1}

        print("[Train]  Start ")
        for epoch in range(self.max_epoch):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("[Evaluate]")
            hit_ratio, gain = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "Hit_Ratio: {}, Gain: {}".format(hit_ratio, gain))
            if hit_ratio > optimized['hit_ratio'] or gain > optimized['gain']:
                optimized['hit_ratio'] = hit_ratio
                optimized['gain'] = gain
                optimized['epoch'] = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - optimized['epoch'] > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best Hit_Ratio: {}, Gain: {}, At Epoch {}".format(optimized['hit_ratio'], optimized['gain'], optimized['epoch']))
        print("[Train] End")


    def run_epoch(self, sess, verbose=10):
        # TODO
        # get shuffled data
        # user_total = norm_train['userId']
        # item_total = norm_train['movieId']
        # rate_total = norm_train['rating']



        # train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        # train_len = len(train_u)
        # shuffled_idx = np.random.permutation(np.arange(train_len))
        # train_u = train_u[shuffled_idx]
        # train_i = train_i[shuffled_idx]
        # train_r = train_r[shuffled_idx]

        num_batches = norm_train.shape[0] // self.batch_size + 1
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batch_size
            max_idx = np.min([train_len, (i+1)*self.batch_size])
            user_batch = user_total[min_idx: max_idx]
            item_batch = item_total[min_idx: max_idx]
            rate_batch = rate_total[min_idx: max_idx]

            feed_dict = {self.user: user_batch, self.item: item_batch, self.rate: rate_batch}
            _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss : {}".format(loss))
        return loss
