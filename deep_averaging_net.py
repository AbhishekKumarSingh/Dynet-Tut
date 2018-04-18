# Deep Averaging Network
# Author: Abhishek Singh
# Paper link: https://aclanthology.info/papers/P15-1162/p15-1162

import dynet as dy

# constants
vocab_size = 10000
emb_dim = 60
h1_dim = #
h2_dim = #

# load
w2i, i2w, i2l, l2i = load("vocab_file.text")

# create model and add parameters
model = dy.Model()
E = model.add_lookup_parameters((vocab_size, emb_dim))
pW1 = model.add_parameters((h1_dim, emb_dim))
pb1 = model.add_parameters(h1_dim)
pW2 = model.add_parameters((h2_dim, h1_dim))
pb2 = model.add_parameters(h2_dim)

trainer = dy.SimpleSGDTrainer(model)

def layer1(x):
    W = dy.parameter(pW1)
    b = dy.parameter(b1)
    return dy.tanh(W*x + b)

def layer2(x):
    W = dy.parameter(pW2)
    b = dy.parameter(b2)
    return dy.tanh(W*x + b)

def encode_doc(doc):
    doc_vec_int = [w2i[word] for word in doc]
    embs = [E[ing] for intg in doc_vec_int]
    return dy.esum(embs)

# incorporating TF/IDF score for word embdeddings
def encode2_doc(doc):
    weights = [tfidf(word) for word in doc]
    doc = [w2i[word] for word in doc]
    embs = [E[intg]*w for w, intg in zip(weights, doc)]

def predict_label(doc):
    x = encode_doc(doc)
    h = layer1(x)
    y = layer2(h)
    return dy.softmax(y)

def compute_loss(prob, label):
    label = l2i[label] # convert label to integer
    return -dy.log(dy.pick(prob, label)) # only pick prob corresponding to label

def classify(doc):
    dy.renew_cg()
    prob = predict_label(doc)
    val = prob.npvalue()
    return i2l[np.argmax(val)]



# cummulative loss
closs = 0.0
for epoch in xrange(1000):
    for (doc, label) in data:
        # renew computation graph
        dy.renew_cg()
        prob = predict_label(doc)
        loss = compute_loss(prob, label)

        closs += loss.scalar_value() # forward pass
            if epoch > 0 and epoch % 1000 == 0:
            print "epoch:", epoch, "loss:", closs/(1000+len(data))
            closs = 0
        loss.backward() # gradient computation
        trainer.update()
