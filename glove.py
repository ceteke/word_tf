import numpy as np
import nltk, operator, sys, pickle, itertools, random
import cudamat as cm
import pymp
pymp.config.nested = True

class Glove:
    def __init__(self, verbose=0):
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.id2tok = {}
        self.tok2id = {}
        self.verbose = verbose
        self.cooccurrence_matrix = None
        self.embedding_matrix = None
        self.ignored_words = ['a', 'the', 'am', 'of', 'and', 'in', 'to', 'is', 's', 'that', 'there', 'not', 'it']

    def __check_fit(self):
        assert len(self.id2tok) > 0 and len(self.tok2id) > 0, "Corpus is not fitted"

    def __check_cooc(self):
        assert self.cooccurrence_matrix is not None, "Co-Occurrence Matrix is not formed"

    def __check_train(self):
        assert self.embedding_matrix is not None, "Training is not done"

    def __form_cooccurence(self, corpus, window_size, vocab_size):
        self.__check_fit()
        cooccurrences = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        num_sentence = len(corpus)
        for s, sentence in enumerate(corpus):
            sent_proc = sentence.lower().strip()
            tokens = self.tokenizer.tokenize(sent_proc)
            sentence_idx = [self.tok2id[t] if t in self.tok2id else -1 for t in tokens]
            num_toks = len(sentence_idx)
            for idx, id in enumerate(sentence_idx):
                if self.verbose == 1:
                    sys.stdout.write("\r" + 'Sentence:{}/{}, Token:{}/{}'.format(s+1,num_sentence,idx+1,num_toks))
                    sys.stdout.flush()
                context_start = max(0, idx-window_size)
                context_end = min(len(sentence_idx)-1, idx+window_size)
                left_context = sentence_idx[context_start:idx]
                right_context = sentence_idx[idx+1:context_end+1]

                for l_idx, l_id in enumerate(left_context):
                    if id != -1:
                        num = 1 / (window_size-l_idx)
                        cooccurrences[id][l_id] += num

                for r_idx, r_id in enumerate(right_context):
                    if id != -1:
                        num = 1 / (r_idx+1)
                        cooccurrences[id][r_id] += num

        return cooccurrences

    def save(self):
        pickle.dump(self, open('glove.pkl', 'wb'), protocol=4)

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

        self.cooccurrence_matrix = self.__form_cooccurence(corpus, window_size, vocab_size)
        self.vocab_size = vocab_size

    def __f(self, x_ij, alpha, x_max):
        result = (x_ij/x_max)**alpha if x_ij < x_max else 1.0
        return result

    def train_iterative(self, embedding_size, learning_rate, epochs, alpha, x_max):
        self.__check_fit()
        self.__check_cooc()

        W = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))
        W_tilda = np.random.normal(-0.1, 0.1, (self.vocab_size, embedding_size))
        b = np.zeros((self.vocab_size))
        b_tilda = np.zeros((self.vocab_size))

        W_grads = np.ones((self.vocab_size, embedding_size))
        W_tilda_grads = np.ones((self.vocab_size, embedding_size))
        b_grads = np.ones((self.vocab_size))
        b_tilda_grads = np.ones((self.vocab_size))

        for e in range(epochs):
            J_total = 0.0

            with pymp.Parallel(8) as p1:
                for i in p1.range(0, self.vocab_size):
                    for j in range(self.vocab_size):
                        x_ij = self.cooccurrence_matrix[i][j]

                        W_i = W[i]
                        W_grad = W_grads[i]
                        b_grad = b_grads[i]

                        with p1.lock:
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

                        with p1.lock:
                            W_tilda[j] -= W_t_update
                            b_tilda[j] -= b_t_update

                        W_grads[i] += np.square(d_W)
                        b_grads[i] += np.square(d_b)

                        with p1.lock:
                            W_tilda_grads[j] += np.square(d_W_tilda)
                            b_tilda_grads[j] += np.square(d_b_tilda)


            print("Error iteration {}: {}".format(e+1, J_total/(self.vocab_size**2)))

        self.embedding_matrix = W + W_tilda


    def train_cuda(self, embedding_size, learning_rate, epochs, alpha, x_max):
        self.__check_fit()
        self.__check_cooc()

        cm.cublas_init()

        f = lambda x: 1.0 if x >= x_max else np.power(x/x_max, alpha)
        f_vec = np.vectorize(f)

        self.cooccurrence_matrix = np.add(self.cooccurrence_matrix, 1e-5)

        W = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))
        W_tilda = np.random.normal(-0.1, 0.1, (self.vocab_size, embedding_size))
        W_grads = np.ones((self.vocab_size, embedding_size))
        W_tilda_grads = np.ones((self.vocab_size, embedding_size))
        lr = np.ones((self.vocab_size, embedding_size)) * learning_rate

        W = cm.CUDAMatrix(W)
        W_grads = cm.CUDAMatrix(W_grads)
        W_tilda = cm.CUDAMatrix(W_tilda)
        W_tilda_grads = cm.CUDAMatrix(W_tilda_grads)
        log_X = cm.CUDAMatrix(np.log(self.cooccurrence_matrix))
        f_X = cm.CUDAMatrix(f_vec(self.cooccurrence_matrix))
        lr = cm.CUDAMatrix(lr)

        for e in range(epochs):
            common = cm.empty((self.vocab_size, self.vocab_size))
            cm.dot(W, W_tilda.transpose()).subtract(log_X, target=common)

            J = cm.empty((self.vocab_size, self.vocab_size))
            cm.pow(common, 2, target=J)
            J.mult(f_X)

            d_common = cm.empty((self.vocab_size, self.vocab_size))
            common.mult(f_X, target=d_common)

            d_W = cm.dot(d_common, W_tilda)
            d_W_tilda = cm.dot(d_common, W)

            d_W_lr = cm.empty((self.vocab_size, embedding_size))
            cm.sqrt(W_grads, target=d_W_lr)
            d_W_t_lr = cm.empty((self.vocab_size, embedding_size))
            cm.sqrt(W_tilda_grads, target=d_W_t_lr)

            lr.divide(d_W_lr, target=d_W_lr)
            d_W_lr.mult(d_W)
            d_W_lr.div_by_scalar(self.vocab_size)

            lr.divide(d_W_t_lr, target=d_W_t_lr)
            d_W_t_lr.mult(d_W_tilda)
            d_W_t_lr.div_by_scalar(self.vocab_size)

            W.subtract(d_W_lr)
            W_tilda.subtract(d_W_t_lr)

            W_grads.add(cm.pow(d_W, 2))
            W_tilda_grads.add(cm.pow(d_W_tilda, 2))

            J_avg = np.sum(J.asarray()*0.5/(self.vocab_size*self.vocab_size))
            print(J_avg)

        self.embedding_matrix = W.asarray() + W_tilda.asarray()

    def most_similar(self, word, n=15):
        word_id = self.tok2id[word]

        dists = np.dot(self.embedding_matrix, self.embedding_matrix[word_id])
        top_ids = np.argsort(dists)[::-1][:n + 1]

        return [self.id2tok[id] for id in top_ids if id != word_id][:n]