# Implementing MLP class in DyNet

import dynet as dy


class MLP(object):
    def __init__(self, model, inp_dim, hid_dim, out_dim, activation=dy.tanh):
        self._W1 = model.add_parameters((hid_dim, inp_dim))
        self._b1 = model.add_parameters((hid_dim))
        self._W2 = model.add_parameters((out_dim, hid_dim))
        self._b2 = model.add_parameters(out_dim)
        self.activation = activation

    def __call__(self, inp_exp):
        W1 = dy.parameter(self._W1)
        b1 = dy.parameter(self._b1)
        W2 = dy.parameter(self._W2)
        b2 = dy.parameter(self._b2)
        f = self.activation
        return f(W2 *f(W1*inp_exp + b1) + b2)


# Test MLP object
x = dy.inputVector(range(10))
model = dy.Model()
mlp = MLP(model, 10, 100, 2)

y = mlp(x)
print y.vec_value()
