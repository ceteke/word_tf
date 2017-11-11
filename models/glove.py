import numpy as np
import nltk, operator, sys, pickle, random
import tensorflow as tf
from sklearn.utils import shuffle

class Glove:
    def __init__(self, verbose=0):
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.id2tok = {}
        self.tok2id = {}
        self.verbose = verbose
        self.cooccurence_dict = {}
        self.embedding_matrix = None
        self.ignored_words = ['a', 'the', 'am', 'of', 'and', 'in', 'to', 'is', 's']

    def __check_fit(self):
        assert len(self.id2tok) > 0 and len(self.tok2id) > 0, "Corpus is not fitted"

    def __check_cooc(self):
        assert len(self.cooccurence_dict) == 0, "Co-Occurrence Matrix is not formed"

    def __check_train(self):
        assert self.embedding_matrix is not None, "Training is not done"

    def __form_cooccurence(self, corpus, window_size, vocab_size):
        self.__check_fit()

        num_sentence = len(corpus)
        for s, sentence in enumerate(corpus):
            sent_proc = sentence.lower().strip()
            tokens = self.tokenizer.tokenize(sent_proc)
            sentence_idx = [self.tok2id[t] if t in self.tok2id else -1 for t in tokens]
            num_toks = len(sentence_idx)
            for idx, id in enumerate(sentence_idx):
                if id == -1:
                    continue
                if self.verbose == 1:
                    sys.stdout.write("\r" + 'Sentence:{}/{}, Token:{}/{}'.format(s+1,num_sentence,idx+1,num_toks))
                    sys.stdout.flush()
                context_start = max(0, idx-window_size)
                context_end = min(len(sentence_idx)-1, idx+window_size)
                left_context = sentence_idx[context_start:idx]
                right_context = sentence_idx[idx+1:context_end+1]

                for l_idx, l_id in enumerate(left_context):
                    if l_id != -1 and l_id != id:
                        num = 1 / (window_size-l_idx)
                        dict_value = self.cooccurence_dict.get((id, l_id), 0.0)
                        self.cooccurence_dict[(id, l_id)] = dict_value + num

                for r_idx, r_id in enumerate(right_context):
                    if r_id != -1 and r_id != id:
                        num = 1 / (r_idx+1)
                        dict_value = self.cooccurence_dict.get((id, r_id), 0.0)
                        self.cooccurence_dict[(id, r_id)] = dict_value + num

    def save(self, name):
        pickle.dump(self, open('glove-{}.pkl'.format(name), 'wb'), protocol=4)

    def save_embedding(self, name):
        assert self.embedding_matrix is not None, "Train first"
        pickle.dump(self.embedding_matrix, open('glove-embedding-{}.pkl'.format(name), 'wb'), protocol=4)

    def fit_corpus(self, corpus, window_size, vocab_size):
        tok2count = {}
        for sentence in corpus:
            sent_proc = sentence.lower().strip()
            tokens = self.tokenizer.tokenize(sent_proc)
            for token in tokens:
                if token not in self.ignored_words:
                    count = tok2count.get(token,0) + 1
                    tok2count[token] = count

        vocab_tokens = dict(sorted(tok2count.items(), key=operator.itemgetter(1), reverse=True)[:vocab_size]).keys()
        tok_idx = 0
        for vt in vocab_tokens:
            self.tok2id[vt] = tok_idx
            self.id2tok[tok_idx] = vt
            tok_idx += 1

        self.__form_cooccurence(corpus, window_size, vocab_size)
        self.vocab_size = vocab_size

    def __f(self, x_ij, alpha, x_max):
        result = (x_ij/x_max)**alpha if x_ij < x_max else 1.0
        return result

    def train_iterative(self, embedding_size, learning_rate, epochs, alpha, x_max):
        self.__check_fit()
        self.__check_cooc()

        W = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))
        W_tilda = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))
        b = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))
        b_tilda = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))

        W_grads = np.ones((self.vocab_size, embedding_size))
        W_tilda_grads = np.ones((self.vocab_size, embedding_size))
        b_grads = np.ones((self.vocab_size))
        b_tilda_grads = np.ones((self.vocab_size))

        for e in range(epochs):
            J_total = 0.0

            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    if (i, j) not in self.cooccurence_dict:
                        continue
                    x_ij = self.cooccurence_dict[(i, j)]

                    W_i = W[i]
                    W_grad = W_grads[i]
                    b_grad = b_grads[i]

                    W_t_j = W[j]
                    W_tilda_grad = W_tilda_grads[j]
                    b_tilda_grad = b_tilda_grads[j]

                    inner = np.dot(W_i, W_t_j) - np.log(x_ij + 1e-60)

                    weight = self.__f(x_ij, alpha, x_max)
                    J = 0.5 * weight * np.square(inner)

                    J_total += J

                    common_term = weight * inner
                    d_W = common_term * W_t_j

                    d_W_tilda = common_term * W_i
                    d_b = d_b_tilda = common_term

                    W[i] -= (learning_rate / np.sqrt(W_grad + 1e-60)) * d_W
                    b[i] -= (learning_rate / np.sqrt(b_grad + 1e-60)) * d_b

                    W_t_update = (learning_rate / np.sqrt(W_tilda_grad + 1e-60)) * d_W_tilda
                    b_t_update = (learning_rate / np.sqrt(b_tilda_grad + 1e-60)) * d_b_tilda

                    W_tilda[j] -= W_t_update
                    b_tilda[j] -= b_t_update

                    W_grads[i] += np.square(d_W)
                    b_grads[i] += np.square(d_b)

                    W_tilda_grads[j] += np.square(d_W_tilda)
                    b_tilda_grads[j] += np.square(d_b_tilda)


            print("Error iteration {}: {}".format(e+1, J_total/(self.vocab_size**2)))

        self.embedding_matrix = W + W_tilda

    def __form_variables(self, embedding_size, batch_size, learning_rate):
        self.left_words = tf.placeholder(tf.int32, shape=[batch_size])
        self.right_words = tf.placeholder(tf.int32, shape=[batch_size])
        self.counts = tf.placeholder(tf.float32, shape=[batch_size])

        self.W = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=1e-3))
        self.W_tilda = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=1e-3))
        self.b = tf.Variable(tf.truncated_normal([self.vocab_size], stddev=1e-3))
        self.b_tilda = tf.Variable(tf.truncated_normal([self.vocab_size], stddev=1e-3))

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    def __form_graph(self, embedding_size, batch_size, learning_rate, alpha, x_max):
        self.__form_variables(embedding_size, batch_size, learning_rate)

        left_embedding = tf.nn.embedding_lookup(self.W, self.left_words)
        right_embedding = tf.nn.embedding_lookup(self.W_tilda, self.right_words)
        left_bias = tf.nn.embedding_lookup(self.b, self.left_words)
        right_bias = tf.nn.embedding_lookup(self.b_tilda, self.left_words)

        weighting_factor = tf.minimum(1.0, tf.pow(tf.div(self.counts,
                                                         tf.constant(x_max, dtype=tf.float32)),
                                                  tf.constant(alpha, dtype=tf.float32)))

        dot_product = tf.reduce_sum(tf.multiply(left_embedding, right_embedding), axis=1)
        log_X = tf.log(self.counts)

        J = tf.square(tf.add_n([dot_product, left_bias, right_bias, -log_X])) * weighting_factor
        J = tf.reduce_sum(J) * 0.5

        train_op = self.optimizer.minimize(J, global_step=self.global_step)
        embedding_op = tf.add(self.W, self.W_tilda)

        return J, train_op, embedding_op

    def __prepare_data(self):
        data = list(self.cooccurence_dict.items())
        i_idxs = []
        j_idxs = []
        counts = []
        for idx, count in data:
            i_idxs.append(idx[0])
            j_idxs.append(idx[1])
            counts.append(count)
        return i_idxs, j_idxs, counts

    def __get_batches(self, data, batch_size):
        limit = (len(data[0])//batch_size)*batch_size
        i_idxs = data[0][0:limit]
        j_idxs = data[1][0:limit]
        counts = data[2][0:limit]

        i_idxs, j_idxs, counts = shuffle(i_idxs, j_idxs, counts)

        for ndx in range(0, limit, batch_size):
            yield i_idxs[ndx:ndx + batch_size], j_idxs[ndx:ndx + batch_size], counts[ndx:ndx + batch_size]

    def train_tf(self, sess, embedding_size, learning_rate, epochs, alpha, x_max, batch_size=512, info_step=1000):
        loss_op, train_op, embedding_op = self.__form_graph(embedding_size, batch_size, learning_rate, alpha, x_max)
        sess.run(tf.global_variables_initializer())

        info_loss = 0.0
        print("Forming training data")
        data = self.__prepare_data()
        self.cooccurence_dict = {} # Free some memory
        print("Started training")
        for e in range(epochs):
            print("Epoch {}/{}".format(e+1, epochs))
            batch_generator = enumerate(self.__get_batches(data, batch_size))
            epoch_loss = 0.0
            for i, b in batch_generator:
                loss, _ = sess.run([loss_op, train_op], feed_dict={self.left_words:b[0], self.right_words:b[1], self.counts:b[2]})
                info_loss += loss
                epoch_loss += loss

                if (i+1) % info_step == 0:
                    print('\tBatch {} loss: {}'.format(i+1, info_loss/info_step))
                    info_loss = 0.0

            print('\tAverage loss: {}'.format(epoch_loss/(len(self.cooccurence_dict) / batch_size)))

        self.embedding_matrix = embedding_op.eval(session=sess)

    def most_similar(self, word, n=15):
        word_id = self.tok2id[word]

        dists = np.dot(self.embedding_matrix, self.embedding_matrix[word_id])
        top_ids = np.argsort(dists)[::-1][:n + 1]

        return [self.id2tok[id] for id in top_ids if id != word_id][:n]