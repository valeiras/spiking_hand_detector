import matplotlib.pyplot as plt
import sklearn.svm

import dvs_loader

[train_x, train_y], [test_x, test_y] = dvs_loader.load_dataset()
# n = len()
# assert len(images) == len(labels)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_x[i], vmin=0, vmax=1, cmap='gray')
    plt.title('label = %d' % train_y[i])

plt.show()

train_x = train_x.reshape(train_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0], -1)


print("Training Linear SVC")
svc = sklearn.svm.LinearSVC()
svc.fit(train_x, train_y)
score = svc.score(test_x, test_y)
print score
