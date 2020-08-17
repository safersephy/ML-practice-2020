function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%basic cost
h = (sigmoid(X * theta));
J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));

%Add regularization
J = J + (lambda / (2 * m) * sum(theta(2:length(theta)) .^ 2));

%basic gradient
grad(1:length(grad)) = (1 / m) * ((h - y)' * X);

%add regularization for all but bias
grad(2:length(grad)) = grad(2:length(grad))' + (lambda / m) * (theta(2:length(theta)));


% =============================================================

end
