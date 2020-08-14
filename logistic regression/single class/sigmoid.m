function g = sigmoid(z)
%SIGMOID Compute sigmoid function

g = zeros(size(z));

g = 1 ./ (1 + exp(-1 * z));

end
