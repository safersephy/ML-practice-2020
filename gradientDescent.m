function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for item = 1:num_iters

  theta = theta - alpha * ((1/m) * ((X * theta)-y)'* X)';

  % Save the cost J in every iteration    
  J_history(item) = computeCost(X, y, theta);

end

end
