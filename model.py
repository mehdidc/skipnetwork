from lasagne import layers, init
from lasagne.nonlinearities import (linear, sigmoid, rectify, very_leaky_rectify, softmax, tanh)
import theano.tensor as T

def build_dense(w=32, h=32, c=1,
				nb_hidden=100,
				nb_outputs=10,
                nb_blocks=3,
                layer_per_block=3):
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hids = []
    l_hid = layers.DenseLayer(l_in, nb_hidden, nonlinearity=rectify, name="hid0")

    for i in range(1, nb_blocks + 1):
    	l_hid_block_first = l_hid
    	l_alpha = layers.DenseLayer(l_hid_block_first, 1, nonlinearity=sigmoid, name="alpha{}".format(i))
    	for j in range(1, layer_per_block + 1):
        	l_hid = layers.DenseLayer(l_hid, nb_hidden, nonlinearity=rectify, name="hid[{}, {}]".format(i, j))
        	hids.append(l_hid)
        l_hid = Gate((l_hid_block_first, l_hid, l_alpha), name="gate{}".format(i)) 
    l_out = layers.DenseLayer(l_hid, nb_outputs, nonlinearity=softmax, name="output")
    return l_out


class Gate(layers.MergeLayer):

	def __init__(self, incomings, **kwargs):
	    super(Gate, self).__init__(incomings, **kwargs)

	def get_output_shape_for(self, input_shapes):
	   return input_shapes[0]

	def get_output_for(self, inputs, **kwargs):
	    input1, input2, alpha = inputs
	    alpha = T.addbroadcast(alpha, 1)
	    return alpha * input1 + (1 - alpha) * input2