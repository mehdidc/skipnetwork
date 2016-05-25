from lasagne import layers, init
from lasagne.nonlinearities import (linear, sigmoid, rectify, very_leaky_rectify, softmax, tanh)
from lasagne.layers import BatchNormLayer
from sample import SimpleBernoulliSampleLayer
import theano.tensor as T

def tanh_lecun(x):
	A =  1.7159
	B = 0.6666
	return A * T.tanh(B * x)

def build_ciresan_4(w=32, h=32, c=1, nb_outputs=10):
	"""

	"""
	nonlin = tanh
	l_in = layers.InputLayer((None, c, w, h), name="input")
	l_hid = layers.DenseLayer(l_in, 2500, nonlinearity=nonlin, W=init.HeUniform(), name="hid1")
	l_hid = layers.DenseLayer(l_hid, 2000, nonlinearity=nonlin, W=init.HeUniform(), name="hid2")
	l_hid = layers.DenseLayer(l_hid, 1000, nonlinearity=nonlin, W=init.HeUniform(), name="hid3")
	l_hid = layers.DenseLayer(l_hid, 1000, nonlinearity=nonlin, W=init.HeUniform(), name="hid3")
	l_hid = layers.DenseLayer(l_hid, 500, nonlinearity=nonlin, W=init.HeUniform(), name="hid4")
	l_out = layers.DenseLayer(l_hid, 10, nonlinearity=softmax, W=init.HeUniform(), name="output")
	return l_out


def build_ciresan_1(w=32, h=32, c=1, nb_outputs=10):
	"""

	"""
	nonlin = tanh
	l_in = layers.InputLayer((None, c, w, h), name="input")
	l_hid = layers.DenseLayer(l_in, 1000, nonlinearity=nonlin, W=init.HeUniform(), name="hid1")
	l_hid = layers.DenseLayer(l_hid, 500, nonlinearity=nonlin, W=init.HeUniform(), name="hid2")
	l_out = layers.DenseLayer(l_hid, 10, nonlinearity=softmax, W=init.HeUniform(), name="output")
	return l_out


def build_dense(w=32, h=32, c=1,
				nb_hidden=100,
				nb_hidden_first=None,
				nb_outputs=10,
                nb_blocks=3,
                layer_per_block=3,
                sample_alpha=False,
                dropout_p=0):
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hids = []
    if nb_hidden_first is None:
    	nb_hidden_first = nb_hidden
    l_hid = layers.DenseLayer(l_in, nb_hidden_first, nonlinearity=rectify, W=init.HeUniform(gain='relu'), name="hid0")

    for i in range(1, nb_blocks + 1):
    	l_hid_block_first = l_hid
    	l_alpha = layers.DenseLayer(l_hid_block_first, 1, nonlinearity=sigmoid, name="alpha{}".format(i), b=init.Constant(-2.))
    	if sample_alpha:
    		l_alpha = SimpleBernoulliSampleLayer(l_alpha)
    	for j in range(1, layer_per_block + 1):
        	l_hid = layers.DenseLayer(l_hid, nb_hidden, nonlinearity=rectify, W=init.HeUniform(gain='relu'), name="hid[{}, {}]".format(i, j))
        	if dropout_p > 0:
        		l_hid = layers.DropoutLayer(l_hid, dropout_p)
        	hids.append(l_hid)
        l_hid = Gate((l_hid_block_first, l_hid, l_alpha), name="gate{}".format(i)) 
    l_out = layers.DenseLayer(l_hid, nb_outputs, nonlinearity=softmax, W=init.HeUniform(gain='relu'), name="output")
    return l_out

def build_dense_resid(w=32, h=32, c=1,
					  nb_hidden=100,
					  nb_hidden_first=None,
					  nb_outputs=10,
		              nb_blocks=3,
		              layer_per_block=3,
		              do_batch_norm=True,
		              dropout_p=0):
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hids = []
    if nb_hidden_first is None:
    	nb_hidden_first = nb_hidden
    l_hid = layers.DenseLayer(l_in, nb_hidden_first, nonlinearity=linear, W=init.HeUniform(gain='relu'), name="hid0")
    l_hid_block_first = l_hid
    if do_batch_norm:
		l_hid = BatchNormLayer(l_hid)
    l_hid = layers.NonlinearityLayer(l_hid, rectify)

    for i in range(1, nb_blocks + 1):
    	for j in range(1, layer_per_block + 1):
        	l_hid = layers.DenseLayer(l_hid, nb_hidden, nonlinearity=linear, W=init.HeUniform(gain='relu'), name="hid[{}, {}]".format(i, j))
        	if do_batch_norm:
        		l_hid = BatchNormLayer(l_hid)
        	if j < layer_per_block:
        		l_hid = layers.NonlinearityLayer(l_hid, rectify)

        	if dropout_p > 0:
        		l_hid = layers.DropoutLayer(l_hid, dropout_p)
        	hids.append(l_hid)
        l_hid = layers.ElemwiseSumLayer((l_hid_block_first, l_hid), name="gate{}".format(i))
        l_hid_block_first = l_hid
        if do_batch_norm:
        	l_hid = BatchNormLayer(l_hid)
        l_hid = layers.NonlinearityLayer(l_hid, rectify)

    l_out = layers.DenseLayer(l_hid, nb_outputs, nonlinearity=softmax, W=init.HeUniform(gain='relu'), name="output")
    return l_out

def build_conv(w=32, h=32, c=1,
				nb_filters=32,
				filter_size=5,
				filter_size_first=None,
				sample_alpha=False,
				nb_outputs=10,
                nb_blocks=3,
                layer_per_block=3,
                pool=True,
                dropout_p=0):
	l_in = layers.InputLayer((None, c, w, h), name="input")

	if filter_size_first is None:
		filter_size_first = filter_size
	hids = []
	l_hid = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(filter_size_first, filter_size_first),
            nonlinearity=rectify,
            name="hid0")
	for i in range(1, nb_blocks + 1):

		l_hid_block_first = l_hid
		l_alpha = layers.DenseLayer(l_hid_block_first, 1, nonlinearity=sigmoid, b=init.Constant(-2.), name="alpha{}".format(i))
    	if sample_alpha:
    		l_alpha = SimpleBernoulliSampleLayer(l_alpha)
	
		for j in range(1, layer_per_block + 1):
			#if j == layer_per_block and pool:
			#	nb_filters_ *= 2
			l_hid = layers.Conv2DLayer(
	            l_hid,
	            num_filters=nb_filters,
	            filter_size=(filter_size, filter_size),
	            nonlinearity=rectify,
	            pad='same',
	            W=init.HeUniform(gain='relu'),
	            name="hid[{}, {}]".format(i, j))        	
			if dropout_p > 0:
				l_hid = layers.DropoutLayer(l_hid, dropout_p)
			print(l_hid.output_shape)
			hids.append(l_hid)
		print(l_hid_block_first.output_shape, l_hid.output_shape)
		l_hid = Gate((l_hid_block_first, l_hid, l_alpha), name="gate{}".format(i)) 
		if pool:
			l_hid = layers.MaxPool2DLayer(l_hid, pool_size=(2, 2))
			#nb_filters *= 2


	l_hid = layers.GlobalPoolLayer(l_hid)
	l_out = layers.DenseLayer(l_hid, nb_outputs, nonlinearity=softmax, W=init.HeUniform(gain='relu'), name="output")
	return l_out


class Gate(layers.MergeLayer):

	def __init__(self, incomings, **kwargs):
	    super(Gate, self).__init__(incomings, **kwargs)

	def get_output_shape_for(self, input_shapes):
		return input_shapes[0]

	def get_output_for(self, inputs, **kwargs):
	    input1, input2, alpha = inputs
	    alpha = T.addbroadcast(alpha, 1)
	    x = ['x'] * (input1.ndim - 2)
	    alpha = alpha.dimshuffle(0, 1, *x)
	    return alpha * input1 + (1 - alpha) * input2