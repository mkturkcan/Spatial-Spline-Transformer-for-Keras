# Spatial Spline Transformer for Keras
#This is a port from the spatial spline transformer in Lasagne for Keras; largely resembles the original in Lasagne. Example script is attached.
### References:
#Original: https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py#L551-L689
#Example: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

import theano
import theano.tensor as T
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).
    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements
    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.
    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X

def _interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


class TPSTransformerLayer(Layer):
    """
    Spatial transformer layer
    The layer applies a thin plate spline transformation [2]_ on the input
    as in [1]_. The thin plate spline transform is determined based on the
    movement of some number of control points. The starting positions for
    these control points are fixed. The output is interpolated with a
    bilinear transformation.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the thin plate spline
        transformation as the x and y coordinates of the destination offsets of
        each control point. The output of the localization network  should
        be a 2D tensor, with shape ``(batch_size, 2 * num_control_points)``
    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.
    control_points : integer
        The number of control points to be used for the thin plate spline
        transformation. These points will be arranged as a grid along the
        image, so the value must be a perfect square. Default is 16.
    precompute_grid : 'auto' or boolean
        Flag to precompute the U function [2]_ for the grid and source
        points. If 'auto', will be set to true as long as the input height
        and width are specified. If true, the U function is computed when the
        layer is constructed for a fixed input shape. If false, grid will be
        computed as part of the Theano computational graph, which is
        substantially slower as this computation scales with
        num_pixels*num_control_points. Default is 'auto'.
    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    .. [2]  Fred L. Bookstein (1989):
            Principal warps: thin-plate splines and the decomposition of
            deformations. IEEE Transactions on
            Pattern Analysis and Machine Intelligence.
            http://doi.org/10.1109/34.24792
    Examples
    --------
    Here, we'll implement an identity transform using a thin plate spline
    transform. First we'll create the destination control point offsets. To
    make everything invariant to the shape of the image, the x and y range
    of the image is normalized to [-1, 1] as in ref [1]_. To replicate an
    identity transform, we'll set the bias to have all offsets be 0. More
    complicated transformations can easily be implemented using different x
    and y offsets (importantly, each control point can have it's own pair of
    offsets).
    >>> import numpy as np
    >>> import lasagne
    >>>
    >>> # Create the network
    >>> # we'll initialize the weights and biases to zero, so it starts
    >>> # as the identity transform (all control point offsets are zero)
    >>> W = b = lasagne.init.Constant(0.0)
    >>>
    >>> # Set the number of points
    >>> num_points = 16
    >>>
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=2*num_points,
    ...                                   W=W, b=b, nonlinearity=None)
    >>> l_trans = lasagne.layers.TPSTransformerLayer(l_in, l_loc,
    ...                                          control_points=num_points)
    """

    def __init__(self, localization_network, downsample_factor=1,
                 control_points=16, precompute_grid='auto', **kwargs):
        super(TPSTransformerLayer, self).__init__(**kwargs)
        self.downsample_factor = as_tuple(downsample_factor, 2)
        self.control_points = control_points
        self.locnet = localization_network
        self.precompute_grid = precompute_grid

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        self.regularizers = self.locnet.regularizers
        self.constraints = self.locnet.constraints
        self.right_mat, self.L_inv, self.source_points, self.out_height, self.out_width = _initialize_tps(
                self.control_points, input_shape, self.downsample_factor,
                self.precompute_grid)
        
    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        return input_shapes

    def call(self, X, mask=None):
        localization_parameters = self.locnet.call(X)
        return _transform_thin_plate_spline(
                localization_parameters, X, self.right_mat, self.L_inv,
                self.source_points, self.out_height, self.out_width,
                self.precompute_grid, self.downsample_factor)


def _transform_thin_plate_spline(
        dest_offsets, input, right_mat, L_inv, source_points, out_height,
        out_width, precompute_grid, downsample_factor):

    num_batch, num_channels, height, width = input.shape
    num_control_points = source_points.shape[1]

    # reshape destination offsets to be (num_batch, 2, num_control_points)
    # and add to source_points
    dest_points = source_points + T.reshape(
            dest_offsets, (num_batch, 2, num_control_points))

    # Solve as in ref [2]
    coefficients = T.dot(dest_points, L_inv[:, 3:].T)

    if precompute_grid:

        # Transform each point on the source grid (image_size x image_size)
        right_mat = T.tile(right_mat.dimshuffle('x', 0, 1), (num_batch, 1, 1))
        transformed_points = T.batched_dot(coefficients, right_mat)

    else:

        # Transformed grid
        out_height = T.cast(height // downsample_factor[0], 'int64')
        out_width = T.cast(width // downsample_factor[1], 'int64')
        orig_grid = _meshgrid(out_height, out_width)
        orig_grid = orig_grid[0:2, :]
        orig_grid = T.tile(orig_grid, (num_batch, 1, 1))

        # Transform each point on the source grid (image_size x image_size)
        transformed_points = _get_transformed_points_tps(
                orig_grid, source_points, coefficients, num_control_points,
                num_batch)

    # Get out new points
    x_transformed = transformed_points[:, 0].flatten()
    y_transformed = transformed_points[:, 1].flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
            input_dim, x_transformed, y_transformed,
            out_height, out_width)

    output = T.reshape(input_transformed,
                       (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _get_transformed_points_tps(new_points, source_points, coefficients,
                                num_points, batch_size):
    """
    Calculates the transformed points' value using the provided coefficients
    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2,
        control_points + 3))
    :param num_points: the number of points
    :return: the x and y coordinates of each transformed point. Shape (
        num_batch, 2, num_to_transform)
    """

    # Calculate the U function for the new point and each source point as in
    # ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance

    # Calculate the squared dist between the new point and the source points
    to_transform = new_points.dimshuffle(0, 'x', 1, 2)
    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))
    r_2 = T.sum(((stacked_transform - source_points.dimshuffle(
            'x', 1, 0, 'x')) ** 2), axis=2)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = T.log(r_2)
    distances = T.switch(T.isnan(log_r_2), r_2 * log_r_2, 0.)

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = T.concatenate([T.ones((batch_size, 1, new_points.shape[2]),
                                        dtype=theano.config.floatX),
                                 new_points], axis=1)
    right_mat = T.concatenate([upper_array, distances], axis=1)

    # Calculate the new value as the dot product
    new_value = T.batched_dot(coefficients, right_mat)
    return new_value


def _U_func_numpy(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)


def _initialize_tps(num_control_points, input_shape, downsample_factor,
                    precompute_grid):
    """
    Initializes the thin plate spline calculation by creating the source
    point array and the inverted L matrix used for calculating the
    transformations as in ref [2]_
    :param num_control_points: the number of control points. Must be a
        perfect square. Points will be used to generate an evenly spaced grid.
    :param input_shape: tuple with 4 elements specifying the input shape
    :param downsample_factor: tuple with 2 elements specifying the
        downsample for the height and width, respectively
    :param precompute_grid: boolean specifying whether to precompute the
        grid matrix
    :return:
        right_mat: shape (num_control_points + 3, out_height*out_width) tensor
        L_inv: shape (num_control_points + 3, num_control_points + 3) tensor
        source_points: shape (2, num_control_points) tensor
        out_height: tensor constant specifying the ouptut height
        out_width: tensor constant specifying the output width
    """

    # break out input_shape
    _, _, height, width = input_shape

    # Create source grid
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
    source_points = np.vstack(
            (x_control_source.flatten(), y_control_source.flatten()))

    # Convert to floatX
    source_points = source_points.astype(theano.config.floatX)

    # Get number of equations
    num_equations = num_control_points + 3

    # Initialize L to be num_equations square matrix
    L = np.zeros((num_equations, num_equations), dtype=theano.config.floatX)

    # Create P matrix components
    L[0, 3:num_equations] = 1.
    L[1:3, 3:num_equations] = source_points
    L[3:num_equations, 0] = 1.
    L[3:num_equations, 1:3] = source_points.T

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = _U_func_numpy(
                    source_points[0, point_1], source_points[1, point_1],
                    source_points[0, point_2], source_points[1, point_2])

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Invert
    L_inv = np.linalg.inv(L)

    if precompute_grid:
        # Construct grid
        out_height = np.array(height // downsample_factor[0]).astype('int64')
        out_width = np.array(width // downsample_factor[1]).astype('int64')
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_width),
                               np.linspace(-1, 1, out_height))
        ones = np.ones(np.prod(x_t.shape))
        orig_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        orig_grid = orig_grid[0:2, :]
        orig_grid = orig_grid.astype(theano.config.floatX)

        # Construct right mat

        # First Calculate the U function for the new point and each source
        # point as in ref [2]
        # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
        # squared distance
        to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
        stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
        stacked_source_points = \
            source_points[:, :, np.newaxis].transpose(1, 0, 2)
        r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)

        # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
        log_r_2 = np.log(r_2)
        log_r_2[np.isinf(log_r_2)] = 0.
        distances = r_2 * log_r_2

        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        upper_array = np.ones(shape=(1, orig_grid.shape[1]),
                              dtype=theano.config.floatX)
        upper_array = np.concatenate([upper_array, orig_grid], axis=0)
        right_mat = np.concatenate([upper_array, distances], axis=0)

        # Convert to tensors
        out_height = T.as_tensor_variable(out_height)
        out_width = T.as_tensor_variable(out_width)
        right_mat = T.as_tensor_variable(right_mat)

    else:
        out_height = None
        out_width = None
        right_mat = None

    # Convert to tensors
    L_inv = T.as_tensor_variable(L_inv)
    source_points = T.as_tensor_variable(source_points)

    return right_mat, L_inv, source_points, out_height, out_width

