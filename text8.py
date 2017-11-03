from glove import Glove
import pickle

with open('text8', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

#glove = pickle.load(open('glove.pkl', 'rb'))
glove = Glove(verbose=1)
glove.fit_corpus(corpus, 10, 5000)
glove.save()
print("Training")
glove.train(embedding_size=100, learning_rate=0.01, epochs=25, alpha=0.75, x_max=100)
#print(glove.most_similar(['were', 'are'], ['was'], topn=50))
print(glove.most_similar('idea'))
