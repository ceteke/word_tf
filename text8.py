import pickle
from models.glove import Glove
import tensorflow as tf

with open('text8', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

glove = pickle.load(open('glove-text8.pkl', 'rb'))
#glove = Glove(verbose=1)
#glove.fit_corpus(corpus, 10, 50000)
#glove.save('text8')

print("Training")
sess = tf.Session()
glove.train_tf(sess, 300, 0.05, 100, 0.75, 100, info_step=10000)
#print(glove.most_similar(['were', 'are'], ['was'], topn=50))
print(glove.most_similar('animal'))
glove.save_embedding('text8')