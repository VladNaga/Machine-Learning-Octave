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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%h=sigmoid(X*theta);
%g=-y' * log(h);
%z=(1-y)' * log(1-h);

%k=theta(2:length(theta));

%J=1/m * sum(g-z) + lambda/(2*m) * sum(k.^2);

L=eye(length(theta));
L(1,1)=0;

h= X * theta;

J=1/(2*m) * (sum((h-y).^2)) + lambda/(2*m) * sum(theta(2:end).^2);

grad = 1/m * X' * (h - y) + lambda/m * L * theta; 



% =========================================================================

grad = grad(:);

end
