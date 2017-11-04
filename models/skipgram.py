import nltk, operator, sys, pickle, random
import tensorflow as tf
from sklearn.utils import shuffle

class SkipGram:
    def __init__(self, verbose=0):
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.id2tok = {}
        self.tok2id = {}
        self.verbose = verbose
        self.embedding_matrix = None
        self.ignored_words = ['a', 'the', 'am', 'of', 'and', 'in', 'to', 'is', 's']
        self.X = []
        self.y = []

    def __form_training_data(self, corpus, window_size, vocab_size):
        num_sentence = len(corpus)
        for s, sentence in enumerate(corpus):
            sent_proc = sentence.lower().strip()
            tokens = self.tokenizer.tokenize(sent_proc)
            sentence_idx = [self.tok2id[t]  if t in self.tok2id else -1 for t in tokens]
            num_toks = len(sentence_idx)
            for idx, id in enumerate(sentence_idx):
                if self.verbose == 1:
                    sys.stdout.write("\r" + 'Sentence:{}/{}, Token:{}/{}'.format(s+1,num_sentence,idx+1,num_toks))
                    sys.stdout.flush()
                context_start = max(0, idx - window_size)
                context_end = min(len(sentence_idx) - 1, idx + window_size)
                left_context = sentence_idx[context_start:idx]
                right_context = sentence_idx[idx + 1:context_end + 1]

                for l_idx, l_id in enumerate(left_context):
                    if l_id != id and l_id != -1:
                        self.X.append(id)
                        self.y.append(l_id)

                for r_idx, r_id in enumerate(right_context):
                    if r_id != id and r_id != -1:
                        self.X.append(id)
                        self.y.append(r_id)

    def fit_corpus(self, corpus, window_size, vocab_size):
        tok2count = {}
        for sentence in corpus:
            sent_proc = sentence.lower().strip()
            tokens = self.tokenizer.tokenize(sent_proc)
            for token in tokens:
                if token not in self.ignored_words:
                    count = tok2count.get(token, 0) + 1
                    tok2count[token] = count

        vocab_tokens = dict(sorted(tok2count.items(), key=operator.itemgetter(1), reverse=True)[:vocab_size]).keys()
        tok_idx = 0
        for vt in vocab_tokens:
            self.tok2id[vt] = tok_idx
            self.id2tok[tok_idx] = vt
            tok_idx += 1

        self.__form_training_data(corpus, window_size, vocab_size)
        self.vocab_size = vocab_size

    def save(self, name):
        pickle.dump(self, open('sgram-{}.pkl'.format(name), 'wb'), protocol=4)

    def save_embedding(self, name):
        assert self.embedding_matrix is not None, "Train first"
        pickle.dump(self.embedding_matrix, open('sgram-embedding-{}.pkl'.format(name), 'wb'), protocol=4)

    def __form_variables(self, embedding_size, batch_size, learning_rate):
        self.target_words = tf.placeholder(tf.int32, shape=[batch_size])
        self.context_words = tf.placeholder(tf.int32, shape=[batch_size])

        self.embedding_tensor = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=1e-3))
        self.W = tf.Variable(tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=1e-3)))
        self.b = tf.Variable(tf.zeros([self.vocab_size]))
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    def __form_graph(self, embedding_size, batch_size, learning_rate, num_samples):
        self.__form_variables(embedding_size, batch_size, learning_rate)

        embeddings = tf.nn.embedding_lookup(self.embedding_tensor, self.target_words)
        J = tf.nn.nce_loss(self.W, self.b, self.context_words, embeddings, num_samples, self.vocab_size)
        train_op = self.optimizer.minimize(J, global_step=self.global_step)

        return J, train_op

    def __get_batches(self, batch_size):
        limit = (len(self.X) // batch_size) * batch_size
        X_b = self.X[0:limit]
        y_b = self.y[0:limit]

        X_b, y_b = shuffle(X_b, y_b)
        for ndx in range(0, limit, batch_size):
            yield X_b[ndx:ndx + batch_size], y_b[ndx:ndx + batch_size]

    def train_tf(self, sess, embedding_size, learning_rate, epochs, num_samples, batch_size=512, info_step=1000):
        loss_op, train_op = self.__form_graph(embedding_size, batch_size, learning_rate, num_samples)
        sess.run(tf.global_variables_initializer())

        info_loss = 0.0
        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            batch_generator = enumerate(self.__get_batches(batch_size))
            epoch_loss = 0.0
            for i, b in batch_generator:
                loss, _ = sess.run([loss_op, train_op],
                                   feed_dict={self.target_words: b[0], self.context_words: b[1]})
                info_loss += loss
                epoch_loss += loss

                if (i + 1) % info_step == 0:
                    print('\tBatch {} loss: {}'.format(i + 1, info_loss / info_step))
                    info_loss = 0.0

            print('\tAverage loss: {}'.format(epoch_loss / (len(self.X) / batch_size)))

        self.embedding_matrix = self.embedding_tensor.eval(session=sess)