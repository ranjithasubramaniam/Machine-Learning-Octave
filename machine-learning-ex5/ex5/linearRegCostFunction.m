function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
f = size(theta,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


prediction = X*theta;
sqrerror = (prediction - y).^2;
regterm=(lambda/(2*m))*(sum(theta([2:f],:).^2));
J =(1/(2*m)*sum(sqrerror))+regterm;



grad(1,:) =(1/m).*(X(:,1)'*((X*theta)-y));
grad([2:f],:) =((1/m).*(X(:,[2:f])'*((X*theta)-y)))+((lambda/m).*theta([2:f],:));







% =========================================================================

grad = grad(:);

end
