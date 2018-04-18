# XOR problem using MLP
import dynet as dy
import random

# dataset
data = [([0, 1], 1),
        ([1, 0], 1),
        ([0, 0], 0),
        ([1, 1], 0)]

# create a model and parameters
model = dy.Model()
pU = model.add_parameters((4, 2))
pb = model.add_parameters(4)
pv = model.add_parameters(4)

trainer = dy.SimpleSGDTrainer(model)
# cummulative loss
closs = 0.0

def predict(x):
    U = dy.parameter(pU)
    v = dy.parameter(pv)
    b = dy.parameter(pb)
    return dy.logistic(dy.dot_product(v, dy.tanh(U * x + b)))

def compute_loss(y_pred, y):
    return - ((y*dy.log(y_pred)) + ((1-y)*dy.log(1-y_pred)))


for epoch in xrange(10000):
    random.shuffle(data)
    for x, y in data:
        dy.renew_cg()
        x = dy.inputVector(x)
        # predict
        yhat = predict(x)
        # loss; loss is scalar as calculated on one instance
        loss = compute_loss(yhat, y)
        closs += loss.scalar_value() # forward calculation

        if epoch > 0 and epoch % 1000 == 0:
            print "epoch:", epoch, "loss:", closs/400
            closs = 0
        loss.backward() # compute gradients
        trainer.update()


yhat = predict(dy.inputVector([1, 1]))
yhat.value()
