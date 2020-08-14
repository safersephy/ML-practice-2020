function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification


%NOTE THIS IS SHITTY, USE THE VECTORIZED VERSION!



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

%backprop

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






% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
