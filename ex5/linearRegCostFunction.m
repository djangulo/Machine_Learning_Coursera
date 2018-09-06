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

thetaReg = theta;
thetaReg(1) = 0;

err = ( (X * theta) - y).^2;

J = (1 / (2*m)) * sum( err ) + ( (lambda / (2*m)) * sum( thetaReg.^2 ) );

% J = (1 / (2 * m)) * sum(((X * theta) - y).^2) + (lambda / (2 * m)) * sum( thetaReg.^2 );

gradErr = ( (X * theta) - y) .* X;

grad = (1 / m) * sum(gradErr) + (lambda / m) .* thetaReg';


% [J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);




% =========================================================================

grad = grad(:);

end
