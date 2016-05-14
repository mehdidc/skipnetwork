from invoke import task
from collections import OrderedDict

from model import build_dense
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates
from lasagnekit.datasets.mnist import MNIST
from helpers import iterate_minibatches
from tabulate import tabulate

@task
def train():
	nb_epochs = 120
	c = 1
	w = 28
	h = 28
	learning_rate = 0.01
	momentum = 0.9
	batchsize = 128

	X = T.tensor4()
	y = T.ivector()
	net = build_dense(
		w=w, h=w, c=c, 
		nb_hidden=50, nb_outputs=10, 
		nb_blocks=4, layer_per_block=3)
	
	print('Compiling the net...')

	y_pred = layers.get_output(net, X)
	y_acc = T.eq(y_pred.argmax(axis=1), y).mean()
	#predict_fn = theano.function([X], y_pred)
	loss = objectives.categorical_crossentropy(y_pred, y)
	loss = loss.mean()
	loss_fn = theano.function([X, y], loss)
	acc_fn = theano.function([X, y], y_acc)

	params = layers.get_all_params(net, trainable=True)
	grad_updates = updates.momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
	train_fn = theano.function([X, y], loss, updates=grad_updates)


	print('Loading data...')

	def preprocess(data):
		return data.reshape((data.shape[0], c, w, h))

	train = MNIST(which='train')
	train.load()
	train.X = preprocess(train.X)
	train.X = train.X[0:128*10]
	train.y = train.y[0:128*10]

	test = MNIST(which='test')
	test.load()
	test.X = preprocess(test.X)


	print('Training...')
	for epoch in range(1, nb_epochs + 1):
		for train_X, train_y in iterate_minibatches(train.X, train.y, batchsize):
			train_fn(train_X, train_y)
		stats = OrderedDict()
		stats['train_loss'] = loss_fn(train.X, train.y)
		stats['test_loss'] = loss_fn(test.X, test.y)
		stats['train_acc'] = acc_fn(train.X, train.y)
		stats['test_acc'] = acc_fn(test.X, test.y)
		stats['epoch'] = epoch
		print(tabulate([stats], headers="keys"))
