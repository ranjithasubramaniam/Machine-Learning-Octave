function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
f = size(theta,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

term1 = y.*log(sigmoid(X*theta));
term2 = (1-y).*log(1-sigmoid(X*theta));
regterm=(lambda/(2*m))*(sum(theta([2:f],:).^2));
J = ((1/m)*sum(-term1-term2))+regterm;

grad(1,:) =(1/m).*(X(:,1)'*(sigmoid(X*theta)-y));
grad([2:f],:) =((1/m).*(X(:,[2:f])'*(sigmoid(X*theta)-y)))+((lambda/m).*theta([2:f],:));



% =============================================================

end