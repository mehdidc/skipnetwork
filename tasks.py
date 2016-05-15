from invoke import task
from collections import OrderedDict
import numpy as np
from model import build_dense, build_conv
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates
from lasagnekit.datasets.mnist import MNIST
from helpers import iterate_minibatches, flip, rotate
from tabulate import tabulate

@task
def train():
	nb_epochs = 120
	c = 1
	w = 28
	h = 28
	learning_rate = theano.shared(np.array(0.1).astype(np.float32))
	momentum = 0.9
	batchsize = 128

	X = T.tensor4()
	y = T.ivector()
	net = build_dense(
	 	w=w, h=w, c=c, 
	 	nb_hidden=500, nb_outputs=10,
	 	nb_blocks=2, layer_per_block=3)

	#net = build_conv(
	#	w=w, h=h, c=c,
	#	nb_filters=16,
	#	filter_size=5,
	#	nb_outputs=10,
	#	nb_blocks=2,
	#	layer_per_block=3,
	#	pool=True
	#)
	
	print('Compiling the net...')

	y_pred = layers.get_output(net, X)
	y_pred_detm = layers.get_output(net, X, deterministic=True)
	#predict_fn = theano.function([X], y_pred)
	
	loss = objectives.categorical_crossentropy(y_pred, y).mean()

	loss_detm = objectives.categorical_crossentropy(y_pred, y).mean()
	y_acc_detm = T.eq(y_pred_detm.argmax(axis=1), y).mean()

	loss_fn = theano.function([X, y], loss_detm)
	acc_fn = theano.function([X, y], y_acc_detm)

	params = layers.get_all_params(net, trainable=True)
	grad_updates = updates.momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
	train_fn = theano.function([X, y], loss, updates=grad_updates)


	print('Loading data...')

	def preprocess(data):
		return data.reshape((data.shape[0], c, w, h))

	train = MNIST(which='train')
	train.load()
	train.X = preprocess(train.X)
	#train.X = train.X[0:128]
	#train.y = train.y[0:128]

	test = MNIST(which='test')
	test.load()
	test.X = preprocess(test.X)


	print('Training...')
	history = []
	for epoch in range(1, nb_epochs + 1):
	    for train_X, train_y in iterate_minibatches(train.X, train.y, batchsize):
	        train_X = rotate(train_X, np.random, min_angle=-45, max_angle=45)
	        train_fn(train_X, train_y)
	    stats = OrderedDict()
	    stats['train_loss'] = loss_fn(train.X, train.y)
	    stats['test_loss'] = loss_fn(test.X, test.y)
	    stats['train_acc'] = acc_fn(train.X, train.y)
	    stats['test_acc'] = acc_fn(test.X, test.y)
	    stats['epoch'] = epoch
	    history.append(stats)
	    print(tabulate([stats], headers="keys"))
	    
	    lr = learning_rate.get_value()
	    lr *= 0.99
	    learning_rate.set_value(np.array(lr).astype(np.float32))