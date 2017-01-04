import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

# get MNIST data
train, test = datasets.get_mnist()

# define train/test dataset
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

# define three layer network
class MLP(Chain):
	def __init__(self, n_units, n_out):
		super(MLP, self).__init__(
			l1 = L.Linear(None, n_units),
			l2 = L.Linear(None, n_units),
			l3 = L.Linear(None, n_out)
		)
		
	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		y = self.l3(h2)
		return y
	
# using pre-defined classifier
model = L.Classifier(MLP(100, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)

# build trainer object
updater = training.StandardUpdater(train_iter, optimizer)
# iterating 20-times
trainer = training.Trainer(updater, (20, 'epoch'), out='result')

# invoke training loop
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
