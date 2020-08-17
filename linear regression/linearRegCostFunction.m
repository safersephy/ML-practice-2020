function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = X * theta;
J = 1 / ( 2 * m) * sum(( h - y ).^2);


%Add regularization
J = J + (lambda / (2 * m) * sum(theta(2:length(theta)) .^ 2)); 

%basic gradient
grad(1:length(grad)) = (1 / m) * ((h - y)' * X);


%add regularization for all but bias, check if we don't need to transpose the basic gradient

grad(2:length(grad)) = grad(2:length(grad)) + (lambda / m) * (theta(2:length(theta)));


% =========================================================================

grad = grad(:);

end
