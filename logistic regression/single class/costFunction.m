function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression

m = length(y); % number of training examples
 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);
J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));

grad = ((1 / m) * (h - y)' * X)'

end
