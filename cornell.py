from models.glove import Glove

with open('movie_lines', 'r', encoding='utf-8') as f:
    corpus = []
    for line in f.readlines():
        sentence = line.split(' +++$+++ ')[-1]
        corpus.append(sentence)

glove = Glove(verbose=1)
glove.fit_corpus(corpus, 10, 50000)
glove.save()