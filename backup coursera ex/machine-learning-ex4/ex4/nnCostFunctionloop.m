function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%add the bias to X/a1
a1 = [ones(m, 1) X];


%calculate z2 with theta1
z2 = a1 * Theta1';

%calculate a2 with z2
a2 = sigmoid(z2);

%add the bias to a2
a2 = [ones(size(z2, 1), 1) a2];

%calculate z3 with theta2
z3 = a2 * Theta2';

%calculate a3 with z3
h = a3 = sigmoid(z3);

%take best prediction of all classes and convert to a classification vector
%[value,index] = max((a3)', [], 2);


%nn cost function

y_matrix = [];
for i = 1:m
    y_matrix = [y_matrix; full(sparse(1:numel(y(i)), y(i), 1, numel(y(i)), num_labels))];
endfor
                           
J = (-y_matrix .* log(a3) - (1 - y_matrix) .* log( 1 - a3 ));
J = (1/m)*sum(sum(J));

% regularization
                     

J = J + (lambda/(2*m)*sum(sum(Theta1(:,2:input_layer_size + 1).^2)));
J = J + (lambda/(2*m)*sum(sum(Theta2(:,2:hidden_layer_size + 1).^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

a1 = [ones(m, 1) X];
y_matrix = [];
Dsum2 = zeros(size(Theta2));
Dsum1 = zeros(size(Theta1));

X = [ones(m, 1) X];

for t = 1:m

    

    %calculate z2 with theta1
    z2 = a1(t,:) * Theta1';

    %calculate a2 with z2
    a2 = sigmoid(z2);
    
    %add the bias to a2
    a2 = [ones(1, 1) a2];

    %calculate z3 with theta2
    z3 = a2 * Theta2';

    %calculate a3 with z3
    h = a3 = sigmoid(z3);



    y_matrix = full(sparse(1:numel(y(t)), y(t), 1, numel(y(t)), num_labels));

    y_matrix =y_matrix';


    %calculate errorterm for output layer

    d3 = a3' - y_matrix;
    d2 = (Theta2' * d3)(2:end).* sigmoidGradient(z2)';

    Dsum1 = Dsum1 + d2 * a1(t,:);
    Dsum2 = Dsum2 + d3 * a2;
    

endfor
                     
 Theta1_grad = 1/m * Dsum1 + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
 Theta2_grad = 1/m * Dsum2 + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
