function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power

%take a single feature and based on p, create polynomials. (note that this code was created for a single feature)


for i = 1:p
  
  X_poly(:,i) = + X.^i;

endfor


end
