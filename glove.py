import numpy as np
import nltk, operator, sys, pickle

class Glove:
    def __init__(self, verbose=0):
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.id2tok = {}
        self.tok2id = {}
        self.verbose = verbose
        self.cooccurrence_matrix = None
        self.embedding_matrix = None

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
        result = np.power(x_ij/x_max, alpha) if x_ij < x_max else 1.0
        return result

    def train(self, embedding_size, learning_rate, epochs, alpha, x_max):
        self.__check_fit()
        self.__check_cooc()
        self.vocab_size = 100
        W = np.random.normal(0.0, 1e-3, (self.vocab_size, embedding_size))
        W_tilda = np.random.normal(0.0, 1e-3, (self.vocab_size, embedding_size))
        b = np.random.normal(0.0, 1e-3, (self.vocab_size))
        b_tilda = np.random.normal(0.0, 1e-3, (self.vocab_size))

        for e in range(epochs):
            J = 0.0
            d_W = np.zeros((self.vocab_size, embedding_size))
            d_W_tilda = np.zeros((self.vocab_size, embedding_size))
            d_b = np.zeros((self.vocab_size))
            d_b_tilda = np.zeros((self.vocab_size))

            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    x_ij = self.cooccurrence_matrix[i][j]
                    inner = np.dot(W[i], W_tilda[j]) + b[i] + b_tilda[j] + np.log(x_ij + 1e-100)
                    weight = self.__f(x_ij, alpha, x_max)
                    J +=  weight * np.square(inner)

                    d_W[i] += weight * W_tilda[j] * inner
                    d_W_tilda[j] += weight * W[i] * inner
                    d_b[i] += weight * inner
                    d_b_tilda[j] += weight * inner

            print(J)

            W -= learning_rate * d_W
            W_tilda -= learning_rate * d_W_tilda
            b -= learning_rate * d_b
            b_tilda -= learning_rate * d_b_tilda

        self.embedding_matrix = W + W_tilda