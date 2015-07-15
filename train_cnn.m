clear all
close all

load training_set

batch_size = 50;

% We get 3/4 for training and 1/4 for testing.
% We want the number of training sets to be divisible by the batch size
nn = round(3*size(imgs, 3)/(4*batch_size))*batch_size;

training_imgs = imgs(:, :, 1:nn);
test_imgs = imgs(:, :, nn+1:end);
training_labels = labels(:, 1:nn);
test_labels = labels(:, nn+1:end);

% Train a 6c-2s-12c-2s Convolutional Neural Network 

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

cnn = cnnsetup(cnn, training_imgs, training_labels);

%% Training
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 20;

cnn = cnntrain(cnn, training_imgs, training_labels, opts);

save cnn
%% Testing
% Test
[er, ~] = cnntest(cnn, training_imgs(:, :, 1:1000), training_labels(:, 1:1000));
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);

[er, ~] = cnntest(cnn, test_imgs(:, :, 1:1000), test_labels(:, 1:1000));
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

