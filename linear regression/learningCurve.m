function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve

m = size(X, 1);


error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i = 1:m
  %first train the model with 1 to i examples
  tmptheta = trainLinearReg(X(1:i,:), y(1:i), lambda);
  %then compute the error based on the thetas from the trained model
  error_train(i) = linearRegCostFunction(X(1:i,:), y(1:i), tmptheta , 0)
  error_val(i) = linearRegCostFunction(Xval, yval, tmptheta, 0)
  
  %note that we don't use regularization in the error check
  
endfor

end
