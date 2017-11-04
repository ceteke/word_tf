from models.glove import Glove
import tensorflow as tf

with open('movie_lines.txt', 'r', encoding='utf-8', errors='ignore') as f:
    corpus = []
    for line in f.readlines():
        sentence = line.split(' +++$+++ ')[-1]
        corpus.append(sentence)

glove = Glove(verbose=1)
glove.fit_corpus(corpus, 10, 40000)
glove.save('cornell')

print("Train")
sess = tf.Session()
glove.train_tf(sess, 300, 0.05, 10, 0.75, 100, 512, 10000)
print(glove.most_similar('love'))
glove.save_embedding('cornell')