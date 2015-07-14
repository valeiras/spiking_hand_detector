import os

import cv2
import numpy as np
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.io.matlab


shape = 192, 192


def load_to_frames(filename, shape=shape, tstep=2000):

    data = scipy.io.matlab.loadmat(filename)
    ts, x, y, p = (data['td_data'][k][0, 0].ravel() for k in ['ts', 'x', 'y', 'p'])
    assert all(np.diff(ts) >= 0), "times not sorted"
    assert x.max() < shape[1] and y.max() < shape[0]

    tmax = ts.max()
    nt = tmax / tstep

    frames = np.zeros((nt,) + shape)

    for ti in range(len(frames)):
        t0, t1 = ti * tstep, (ti + 1) * tstep
        t0, t1 = np.array([t0, t1], dtype=ts.dtype)
        i0, i1 = np.searchsorted(ts, [t0, t1])

        m = slice(i0, i1)
        tt, xx, yy, pp = ts[m], x[m], y[m], p[m]
        frames[ti, yy, xx] = pp

    return frames


def load_boxes(filename):
    boxes = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            t, x, y = (float(s) for s in line.split(' '))
            boxes.append((t, x, y))

    return boxes


def remove_bad_pixels(frames, threshold=0.1):
    # remove always on and always off pixels
    fmean = frames.mean(0)
    m = (abs(fmean) > threshold)
    frames[:, m] = 0


def filter_frames(frames, alpha=0.1):
    fframe = np.zeros(shape)
    for i in range(len(frames)):
        fframe += alpha * (frames[i] - fframe)
        frames[i] = fframe


def get_distractor_boxes(frame, box_xy, box_shape=(64, 64), acount=100, n=3, rng=np.random):
    box_radius = np.array(box_shape) / 2

    # i = 0
    boxes = []

    # distractor boxes
    for _ in range(100):
        r0, r1 = 2 * rng.randint(0, 2, size=2) - 1
        u = rng.uniform(-1, 1)
        if r0 > 0:
            # fix x, randomize y
            x = box_xy[0] + r1 * (box_radius[0] + 10)
            y = box_xy[1] + u * box_radius[1]
        else:
            # fix y, randomize x
            y = box_xy[1] + r1 * (box_radius[1] + 10)
            x = box_xy[0] + u * box_radius[0]

        dbox_xy = np.array((x, y))
        x0, y0 = (dbox_xy - box_radius).astype(int)
        x1, y1 = (dbox_xy + box_radius).astype(int)

        if x0 < 0 or y0 < 0 or x1 >= shape[1] or y1 >= shape[0]:
            continue  # out of bounds

        dbox = frame[y0:y1, x0:x1]
        if dbox.shape == tuple(box_shape) and abs(dbox).sum() > acount:
            boxes.append(((x0, y0), (x1, y1)))
            if len(boxes) >= n:
                break

    return boxes


def make_images():
    tstep = 2000

    filename = 'data/dvs/andreas_cup.mat'
    frames = load_to_frames(filename, tstep=tstep)
    remove_bad_pixels(frames)
    filter_frames(frames)
    shape = frames.shape[1:]

    box_filename = 'data/dvs/andreas_cup_tracker_verified.txt'
    boxes = load_boxes(box_filename)
    boxes = np.asarray(boxes)
    box_fxy = scipy.interpolate.interp1d(boxes[:, 0], boxes[:, 1:], axis=0, bounds_error=False)

    point = lambda x, y: (int(x), int(y))

    image = np.zeros(shape + (3,), dtype=np.uint8)

    rng = np.random.RandomState(3)

    disti = 0
    nn = 5
    for i in range(len(frames) / nn):
        frame = frames[nn*i:nn*(i+1)].sum(0)
        image[...] = np.clip(128 + 128*frame, 0, 255)[:, :, None]

        ti = (i + 0.5) * tstep * nn
        box_xy = box_fxy(ti)
        box_shape = np.array((64, 64))
        box_radius = box_shape / 2

        if not any(np.isnan(box_xy)):
            x0, y0 = (box_xy - box_shape/2).astype(int)
            x1, y1 = (box_xy + box_shape/2).astype(int)
            box = image[y0:y1, x0:x1]

            if box.shape[:2] == tuple(box_shape):
                cv2.imwrite('data/dvs/images/andreas_cup_hand_%d.png' % i, box)
                print("Wrote frame")

            dboxes = get_distractor_boxes(
                frame, box_xy, box_shape=box_shape, acount=100, n=2, rng=rng)
            for [x0, y0], [x1, y1] in dboxes:
                dbox = image[y0:y1, x0:x1]
                assert dbox.shape[:2] == tuple(box_shape)
                cv2.imwrite('data/dvs/images/andreas_cup_dist_%d.png' % disti,
                            dbox)
                print("Wrote distractor")
                disti += 1


def test_video():
    import matplotlib.pyplot as plt
    plt.ion()

    tstep = 2000

    filename = 'data/dvs/andreas_cup.mat'
    frames = load_to_frames(filename, tstep=tstep)
    remove_bad_pixels(frames)
    filter_frames(frames, alpha=0.2)
    shape = frames.shape[1:]

    box_filename = 'data/dvs/andreas_cup_tracker_verified.txt'
    boxes = load_boxes(box_filename)
    boxes = np.asarray(boxes)
    box_fxy = scipy.interpolate.interp1d(boxes[:, 0], boxes[:, 1:], axis=0, bounds_error=False)

    print("Video...")
    plt.figure(1)
    plt.clf()
    image = np.zeros(shape + (3,), dtype=np.uint8)
    img = plt.imshow(image)
    # img = plt.imshow(image, vmin=-1, vmax=1, cmap='gray')

    point = lambda x, y: (int(x), int(y))

    rng = np.random.RandomState(3)

    nn = 5
    for i in range(len(frames) / nn):
        frame = frames[nn*i:nn*(i+1)].sum(0)
        image[...] = np.clip(128 + 128*frame, 0, 255)[:, :, None]

        ti = (i + 0.5) * tstep * nn
        box_xy = box_fxy(ti)
        box_shape = np.array((64, 64))
        box_radius = box_shape / 2

        if not any(np.isnan(box_xy)):
            cv2.rectangle(image,
                          point(*(box_xy - box_shape/2)),
                          point(*(box_xy + box_shape/2)),
                          (255, 0, 0))
            dboxes = get_distractor_boxes(
                frame, box_xy, box_shape=box_shape, acount=100, n=3, rng=rng)

            for point0, point1 in dboxes:
                cv2.rectangle(image, point0, point1, (0, 0, 255))

        # img.set_data(frames[10*i:10*i+10].sum(0))
        # img.set_data(frames[10*i])
        img.set_data(image)
        plt.draw()


def load_dataset(shuffle=True, rng=np.random):
    img_dir = 'data/dvs/images'
    filelist = os.listdir(img_dir)
    filelist = filter(lambda s: s.endswith('.png'), filelist)

    n = len(filelist)
    images = np.zeros((n, 64, 64))
    labels = np.zeros(n)
    for i, filename in enumerate(filelist):
        png = cv2.imread(os.path.join(img_dir, filename))
        images[i] = png.mean(2) / 255.
        labels[i] = 1 if 'dist' in filename else 0

    # separate into training set and test set
    ii = [(labels == i).nonzero() for i in np.unique(labels)]
    n = np.min([len(i) for i in ii])  # equal category sizes
    n_train = int(0.8 * n)
    train_x = np.vstack([images[i[:n_train]] for i in ii])
    train_y = np.hstack([labels[i[:n_train]] for i in ii])
    test_x = np.vstack([images[i[n_train:n]] for i in ii])
    test_y = np.hstack([labels[i[n_train:n]] for i in ii])
    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)

    if shuffle:
        def shuffle_set(x, y):
            i = rng.permutation(len(x))
            x[:] = x[i]
            y[:] = y[i]

        shuffle_set(train_x, train_y)
        shuffle_set(test_x, test_y)

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    make_images()
    # test_video()
