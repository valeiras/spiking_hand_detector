import sys
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32, exception_verbosity=high'
import theano
dtype = theano.config.floatX

import dvs_loader
from dvs_loader import load_to_frames, filter_frames
from train_convnet import ConvLayer, FullLayer, Convnet

shape = (192, 192)

# load convnet
layers = [
    ConvLayer(filters=32, filter_size=9, pooling=3, initW=0.01),
    FullLayer(outputs=2, initW=0.01, initB=0., wc=0.01),
    ]

convfile = 'convnet_2015-07-15_20-56-16.npz'
convdata = np.load(convfile)

net = Convnet((1,) + shape, layers)
propup = net.get_propup()

for i, [w, b] in enumerate(zip(convdata['weights'], convdata['biases'])):
    net.layers[i].weights.set_value(w)
    net.layers[i].biases.set_value(b)

if 0:
    # test
    _, [test_x, test_y] = dvs_loader.load_dataset()
    test_x, test_y = test_x[:1000], test_y[:1000]

    test_x = test_x[:, None, :, :].astype(dtype)
    test_y = test_y.astype(np.int32)

    _, test = net.get_train()
    y = test(test_x, test_y)
    print y.mean()

    y = propup(test_x)
    print (test_y != y).mean()

    sys.exit(0)

# load data
filename = 'data/dvs/andreas_cup.mat'
tstep = 1000
frames = load_to_frames(filename, shape=shape, tstep=tstep)
filter_frames(frames)
np.clip(5 * frames, -1, 1, out=frames)

cshape = (60, 60)
s = 3  # skip

plt.figure(1)
plt.clf()

plt.subplot(2, 1, 1)
image = np.zeros(shape + (3,), dtype=np.uint8)
img = plt.imshow(image)

plt.subplot(2, 1, 2)
class_image = -np.ones(((shape[0] - cshape[0] + s) / s, (shape[1] - cshape[1] + s) / s))
class_img = plt.imshow(class_image, vmin=-1, vmax=1)
print class_image.shape

for frame in frames:
    integral, _ = cv2.integral2(abs(frame))
    counts = (integral[:-cshape[0], :-cshape[1]] - integral[cshape[0]:, :-cshape[1]]
              - integral[:-cshape[0], cshape[1]:] + integral[cshape[0]:, cshape[1]:])

    ii, jj = (counts[::s, ::s] > 5).nonzero()
    patches = [frame[s*i:s*i+cshape[0], s*j:s*j+cshape[1]] for i, j in zip(ii, jj)]
    patches = np.array(patches, dtype=dtype)[:, None, :, :]
    classes = propup(patches)

    class_image[:] = -1
    class_image[ii, jj] = classes
    class_img.set_data(class_image)

    image[:] = (127 + frame * 127)[:, :, None]
    img.set_data(image)
    plt.draw()
