# RNN based Language Model

import dynet as dy
import numpy as np

# constants
vocab_size = 7
emb_dim = 64

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

E = model.add_lookup_parameters((vocab_size, emb_dim))
# word level LSTM (layer=1, input=64, hidden=32, model)
RNN = dy.LSTMBuilder(1, 64, 32, model)

# Softmax weights and biases on top of LSTM outputs
W_sm = model.add_parameters((vocab_size, 32))
b_sm = model.add_parameters(vocab_size)

# build the language model graph
def compute_lm_loss(wids):
    dy.renew_cg()
    # parameters -> expressions
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)
    # add parameters to CG and get state
    f_init = RNN.initial_state()

    # get the word vectors for each word ID
    wembs = [E[wid] for wid in wids]

    # start the rnn by inputting "<s>"
    s = f_init.add_input(wembs[-1])
    # process each word ID and embedding
    losses = []
    for wid, we in zip(wids, wembs):
        # calculate and save the softamx loss
        score = W * s.output() + b
        loss = dy.pickneglogsoftmax(score, wid)
        losses.append(loss)

        # update the RNN state with input
        s = s.add_input(we)
    # return the sum of all losses
    return dy.esum(losses)


# Dummy text
# <s> the cat sat on the mat </s>
# word ids: 6, 0, 1, 2, 3, 0, 4, 5
wids = [6, 0, 1, 2, 3, 0, 4, 5]
i2word = {6: '<s>', 0: 'the', 1: 'cat', 2:'sat', 3: 'on', 4: 'mat', 5: '</s>'}

# train
closs = 0.0
for epoch in xrange(1000):
    j = compute_lm_loss(wids)
    closs += j.scalar_value() # forward pass
    if epoch > 0 and epoch % 100 == 0:
        print "epoch:", epoch, "loss:", closs/(100)
        closs = 0
    j.backward()
    trainer.update()

# test output
def generate_output():
    sent = []
    dy.renew_cg()
    # parameters -> expressions
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)
    # add parameters to CG and get state
    f_init = RNN.initial_state()

    # start the rnn by inputting "<s>"
    s = f_init.add_input(E[6])
    while (True):
        opt = dy.softmax(W * s.output() + b).npvalue()
        # print opt
        w_id = np.argmax(opt)
        # print w_id
        if (len(sent) > 8) or (w_id == 5):
            return sent
        sent.append(i2word[w_id])
        s = s.add_input(E[w_id])

print "Output:", " ".join(generate_output())
