function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

prob = sigmoid(X*theta);

for i = 1:m
    if prob(i,:) >= 0.5
        p(i,:) = 1;
    else
        p(i,:) = 0;
    end
end

end
