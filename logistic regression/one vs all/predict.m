function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network


m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

%feedforward time

%add the bias to X/a1
X = [ones(m, 1) X];
a1 = X;

%calculate z2 with theta1
z2 = Theta1 * a1';

%calculate a2 with z2
a2 = sigmoid(z2);

%add the bias to a2
a2 = [ones(1,m); a2];

%calculate z3 with theta2
z3 = Theta2 * a2;

%calculate a3 with z3
a3 = sigmoid(z3);q

%take best prediction of all classes and convert to a classification vector
[value,index] = max((a3)', [], 2);

p = index;




end
