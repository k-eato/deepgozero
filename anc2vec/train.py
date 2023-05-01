import anc2vec
import anc2vec.train as builder
import pickle

es = builder.fit('../data/go.obo', embedding_sz=1024, batch_sz=64, num_epochs=2)
with open('go_emb_anc2vec.pkl', 'wb') as f:
    pickle.dump(es, f)