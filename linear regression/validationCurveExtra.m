function [lambda_vec, error_train, error_val, error_test] = ...
    validationCurve(X, y, Xval, yval, Xtest, ytest)

%same as regular validationCurve but added with a testset

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_test = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  tmptheta = trainLinearReg(X, y, lambda); 
  error_train(i) = linearRegCostFunction(X, y, tmptheta , 0);
  error_val(i) = linearRegCostFunction(Xval, yval, tmptheta, 0);
  error_test(i) = linearRegCostFunction(Xtest, ytest, tmptheta, 0);
endfor


end
