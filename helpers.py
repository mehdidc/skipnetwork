import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import SimilarityTransform, warp

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def flip(X, random_state=None, direction='vertical', ratio=0.5):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb = X.shape[0]
    X = X.copy()
    indices = rng.choice(nb, int(nb * ratio), replace=False)
    if direction == 'vertical':
        X[indices] = X[indices, :, ::-1, :]
    elif direction == 'horizontal':
        X[indices] = X[indices, ::-1, :, :]
    else:
        raise Exception('unknown direction : {}'.format(direction))
    return X

def rotate_scale(X, min_angle=-15, max_angle=15, min_scale=0.85, max_scale=1.15, random_state=None, inplace=False):
    """
    default values of parameters from Ciresan et al.
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    if inplace:
        X_rot = X
    else:
        X_rot = np.zeros_like(X)
    for i in np.arange(X.shape[0]):
        angle = rng.uniform(min_angle, max_angle)
        scale = rng.uniform(min_scale, max_scale)
        X_rot[i] = rotate_scale_one(X[i], angle, scale)
    return X_rot

def rotate_scale_one(img, angle, scale):
    tform = SimilarityTransform(rotation=np.pi*angle/180., scale=scale)
    return warp(img, tform)


def elastic_transform(X, min_alpha=36, max_alpha=38, min_sigma=5, max_sigma=6, random_state=None, inplace=False):
    """
    default values of parameters from Ciresan et al.
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)

    if inplace:
        X_elas = X
    else:
        X_elas = np.zeros_like(X)
    for i in np.arange(X.shape[0]):
        alpha = rng.uniform(min_alpha, max_alpha)
        sigma = rng.uniform(min_sigma, max_sigma)
        X_elas[i] = elastic_transform_one(
            X[i],
            alpha,
            sigma, 
            rng=rng)
    return X_elas


def elastic_transform_one(image, alpha, sigma, rng=np.random):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    shape = image.shape
    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


