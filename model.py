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
        self.learning_rate = 0.0001
        self.max_epoch = 50
        self.batch_size = 256
        self.topK = 10
        # initialize layers
        self.add_embedding_layer()
        self.add_loss()
        self.add_train_step()
        self.check_point = './check_point'
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
            user_W1 = init_variable([self.shape[1], self.user_layer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.user_layer)-1):
                W = init_variable([self.user_layer[i], self.user_layer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.user_layer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("item_layer"):
            item_W1 = init_variable([self.shape[0], self.item_layer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.item_layer)-1):
                W = init_variable([self.item_layer[i], self.item_layer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.item_layer[i+1]], "item_b"+str(i+2))
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
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)


    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.fit(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if os.path.exists(self.check_point):
            [os.remove(f) for f in os.listdir(self.check_point)]
        else:
            os.mkdir(self.check_point)


   def fit(self):
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
                self.saver.save(self.sess, self.check_point)
            if epoch - optimized['epoch'] > self.early_stop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best Hit_Ratio: {}, Gain: {}, At Epoch {}".format(optimized['hit_ratio'], optimized['gain'], optimized['epoch']))
        print("[Train] End")


    def run_epoch(self, sess, verbose=10):
        user_total = norm_train['userId'].to_numpy()
        iter_total = norm_train['movieId'].to_numpy()
        rate_total = norm_train['rating'].to_numpy()

        num_batches = norm_train.shape[0] // self.batch_size + 1
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batch_size
            max_idx = np.min([len(user_total), (i+1)*self.batch_size])
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
        print("\n Mean loss : {}".format(loss))
        return loss

    def evaluate(self, sess, topK):
        def get_hit_ratio(rank_list, target_item):
            for item in rank_list:
                if item == target_item:
                    return 1
            return 0
        def get_normed_gain(rank_list, target_item):
            for i in range(len(rank_list)):
                item = rank_list[i]
                if item == target_item:
                    return math.log(2) / math.log(i+2)
            return 0

        hit_ratios =[]
        normed_gains = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            rank_list = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            hit_ratio = get_hit_ratio(rank_list, target)
            normed_gain = get_normed_gain(rank_list, target)
            hit_ratios.append(hit_ratio)
            normed_gains.append(normed_gains)
        return np.mean(hit_ratios), np.mean(normed_gains)


    def predict(self, sess):
        # input : one-hot endcoded user & item
        # restore check point saved model
        saver.restore(sess, self.check_point)

        feed_dict = {self.user: user_batch, self.item: item_batch, self.rate: None}
        predict, losses = sess.run(self.y_, feed_dict=feed_dict)
        print("prediction : {} ".format(predict))


if __init__ == 'main':
