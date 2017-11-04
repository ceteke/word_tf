import pickle, sys
from models import glove
import tensorflow as tf
sys.modules['glove'] = glove

with open('text8', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

gloveO = pickle.load(open('glove-text8.pkl', 'rb'))
#glove = Glove(verbose=1)
#glove.fit_corpus(corpus, 10, 50000)
#glove.save('text8')

print("Training")
sess = tf.Session()
gloveO.train_tf(sess, 300, 0.05, 100, 0.75, 100, info_step=10000)
#print(glove.most_similar(['were', 'are'], ['was'], topn=50))
print(gloveO.most_similar('animal'))
gloveO.save_embedding('text8')