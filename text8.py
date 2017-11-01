from glove import Glove
import pickle

with open('text8', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

#glove = pickle.load(open('glove.pkl', 'rb'))
glove = Glove(verbose=1)
print("Fitting corpus...")
glove.fit_corpus(corpus, 10, 100)
glove.save()
glove.train(100, 0.05, 25, 0.75, 100)
#print(glove.most_similar(['were', 'are'], ['was'], topn=50))


