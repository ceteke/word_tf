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
        self.ignored_words = ['a', 'the', 'am', 'of', 'and', 'in', 'to', 'is', 's',]

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

    def train(self, embedding_size, learning_rate, epochs, alpha, x_max):
        self.__check_fit()
        self.__check_cooc()

        W = np.random.uniform(-0.1, 0.1, (self.vocab_size, embedding_size))
        W_tilda = np.random.normal(-0.1, 0.1, (self.vocab_size, embedding_size))

        for e in range(epochs):
            J_total = 0.0
            W_grads = np.ones((self.vocab_size, embedding_size))
            W_tilda_grads = np.ones((self.vocab_size, embedding_size))

            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    x_ij = self.cooccurrence_matrix[i][j]
                    inner = np.dot(W[i], W_tilda[j]) - np.log(x_ij + 1e-60)

                    weight = self.__f(x_ij, alpha, x_max)
                    J = 0.5 * weight * np.square(inner)
                    J_total += J

                    d_W = weight * W_tilda[j] * inner
                    d_W_tilda = weight * W[i] * inner

                    W_grads[i] += np.square(d_W)
                    W_tilda_grads[j] += np.square(d_W_tilda)

                    W[i] -= (learning_rate / np.sqrt(W_grads[i] + 1e-60)) * d_W
                    W_tilda[j] -= (learning_rate / np.sqrt(W_tilda_grads[j] + 1e-60)) * d_W_tilda

            print("Error iteration {}: {}".format(e+1, J_total/(self.vocab_size**2)))

        self.embedding_matrix = W + W_tilda

    def most_similar(self, word, n=15):
        word_id = self.tok2id[word]

        dists = np.dot(self.embedding_matrix, self.embedding_matrix[word_id])
        top_ids = np.argsort(dists)[::-1][:n + 1]

        return [self.id2tok[id] for id in top_ids if id != word_id][:n]