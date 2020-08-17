function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda

%create a number of different lambda values
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';


%create error vectors, with elements for each possible lambda
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


%loop through all lambdas and check the error depending
for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  tmptheta = trainLinearReg(X, y, lambda); 
  error_train(i) = linearRegCostFunction(X, y, tmptheta , 0);
  error_val(i) = linearRegCostFunction(Xval, yval, tmptheta, 0);
endfor


end
