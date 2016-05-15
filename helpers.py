import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import SimilarityTransform, warp
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory

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

def rotate_scale(X, min_angle=-15, max_angle=15, min_scale=0.85, max_scale=1.15, random_state=None, n_jobs=1):
    """
    default values of parameters from Ciresan et al.
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    angles = rng.uniform(min_angle, max_angle, size=X.shape[0])
    scales = rng.uniform(min_scale, max_scale, size=X.shape[0])    
    X_rot = Parallel(n_jobs=n_jobs)(delayed(rotate_scale_one)(X[i], angles[i], scales[i]) for i in range(X.shape[0]))
    return np.array(X_rot, dtype='float32')

def rotate_scale_one(img, angle, scale):
    tform = SimilarityTransform(rotation=np.pi*angle/180., scale=scale)
    return warp(img, tform)


def elastic_transform(X, min_alpha=36, max_alpha=38, min_sigma=5, max_sigma=6, random_state=None, n_jobs=1):
    """
    default values of parameters from Ciresan et al.
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    alphas = rng.uniform(min_alpha, max_alpha, size=X.shape[0])
    sigmas = rng.uniform(min_sigma, max_sigma, size=X.shape[0])
    X_elas = Parallel(n_jobs=n_jobs)(delayed(elastic_transform_one)(X[i], alphas[i], sigmas[i]) for i in range(X.shape[0]))
    return np.array(X_elas, dtype='float32')


def elastic_transform_one(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.randint(0, 999999)
    rng = np.random.RandomState(random_state)
    shape = image.shape
    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


