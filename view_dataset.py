import matplotlib.pyplot as plt
import numpy as np

import dvs_loader

[train_x, train_y], [test_x, test_y] = dvs_loader.load_dataset()

y = np.random.choice(np.unique(test_y), size=len(test_y))
errors = y != test_y
print "Random error: %0.3f" % (errors.mean())

r, c = 5, 5
for i in range(r * c):
    plt.subplot(r, c, i+1)
    plt.imshow(train_x[i], vmin=-1, vmax=1, cmap='gray')
    plt.title('label = %d' % train_y[i])

plt.show()
