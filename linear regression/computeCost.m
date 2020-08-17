function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression

m = length(y); % number of training examples

J = 0;

h = X * theta;

J = 1 / ( 2 * m) * sum(( h - y ).^2);

end
