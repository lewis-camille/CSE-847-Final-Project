% CSE 847 Final Project
% Main Program

% ***********************************************************
% Obtaining and creating training data
% ***********************************************************
% Import training data
data = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% Perform clustering to obtain alternate sets of labels
% K-Means Clustering
kmeans_labels = kmeans(transpose(data), 10, 'MaxIter',1000);

% K-Medoids Clustering
kmedoids_labels = kmedoids(transpose(data), 10, 'Options', statset('MaxIter', 1000));

% Compare clustering labels with true training labels
kmeans_FMI = FMI(kmeans_labels, labels)
kmedoids_FMI = FMI(kmedoids_labels, labels)


% ***********************************************************
% Training
% ***********************************************************

% Need one-hot encodings for each set of labels
one_hot_labels = create_one_hot_encoding(labels);
one_hot_kmeans_labels = create_one_hot_encoding(kmeans_labels);
one_hot_kmedoids_labels = create_one_hot_encoding(kmedoids_labels);

% Train 3 Neural Network models, one for each set of labels
net = patternnet([10, 8, 5]);
net = configure(net, data, one_hot_labels);
net = train(net, data, one_hot_labels);
nn_pred = net(data);
nn_pred = get_labels_from_one_hot(nn_pred);
nn_FMI = FMI(nn_pred, labels)

kmeans_net = patternnet([10, 8, 5]);
kmeans_net = configure(kmeans_net, data, one_hot_kmeans_labels);
kmeans_net = train(kmeans_net, data, one_hot_kmeans_labels);
nn_kmeans_pred = kmeans_net(data);
nn_kmeans_pred = get_labels_from_one_hot(nn_kmeans_pred);
nn_kmeans_FMI = FMI(nn_kmeans_pred, labels)

kmedoids_net = patternnet([10, 8, 5]);
kmedoids_net = configure(kmedoids_net, data, one_hot_kmedoids_labels);
kmedoids_net = train(kmedoids_net, data, one_hot_kmedoids_labels);
nn_kmedoids_pred = kmedoids_net(data);
nn_kmedoids_pred = get_labels_from_one_hot(nn_kmedoids_pred);
nn_kmedoids_FMI = FMI(nn_kmedoids_pred, labels)

% Train 3 Support Vector Machine models, one for each set of labels
svm = fitcecoc(transpose(data), labels);
svm_pred = predict(svm, transpose(data));
svm_FMI = FMI(svm_pred, labels)

svm_kmeans = fitcecoc(transpose(data), kmeans_labels);
svm_kmeans_pred = predict(svm_kmeans, transpose(data));
svm_kmeans_FMI = FMI(svm_kmeans_pred, labels)

svm_kmedoids = fitcecoc(transpose(data), kmedoids_labels);
svm_kmedoids_pred = predict(svm_kmedoids, transpose(data));
svm_kmedoids_FMI = FMI(svm_kmedoids_pred, labels)

% ***********************************************************
% Testing and Evaluation
% ***********************************************************
% Import testing data
data = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Test Neural Network models
nn_pred = net(data);
nn_pred = get_labels_from_one_hot(nn_pred);
nn_FMI_test = FMI(nn_pred, labels)

nn_kmeans_pred = kmeans_net(data);
nn_kmeans_pred = get_labels_from_one_hot(nn_kmeans_pred);
nn_kmeans_FMI_test = FMI(nn_kmeans_pred, labels)

nn_kmedoids_pred = kmedoids_net(data);
nn_kmedoids_pred = get_labels_from_one_hot(nn_kmedoids_pred);
nn_kmedoids_FMI_test = FMI(nn_kmedoids_pred, labels)

% Test Support Vector Machine models
svm_pred = predict(svm, transpose(data));
svm_FMI_test = FMI(svm_pred, labels)

svm_kmeans_pred = predict(svm_kmeans, transpose(data));
svm_kmeans_FMI_test = FMI(svm_kmeans_pred, labels)

svm_kmedoids_pred = predict(svm_kmedoids, transpose(data));
svm_kmedoids_FMI_test = FMI(svm_kmedoids_pred, labels)

function one_hot_labels = create_one_hot_encoding(L)
dimensions = size(L);
one_hot_labels = zeros(10, dimensions(1));
for i = 1:dimensions(1)
    one_hot_labels(L(i) + 1, i) = 1;
end
end

function L = get_labels_from_one_hot(one_hot_labels)
dimensions = size(one_hot_labels);
L = zeros(dimensions(2), 1);
for iClass = 1:dimensions(1)
    for iPt = 1:dimensions(2)
        if one_hot_labels(iClass, iPt) == 1
            L(iPt) = iClass - 1;
        end
    end
end
end
