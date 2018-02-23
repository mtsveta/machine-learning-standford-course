function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(size(theta, 1), 1);

% The cost of a particular choice of theta
J    = - 1/m * (y' * log(sigmoid(X * theta)) + (1 - y') * log(1 - sigmoid(X * theta))) ...
       + lambda / (2 * m) * sum(theta(2:end).^2);
   
% The partial derivatives of the cost w.r.t. each parameter in theta
grad(1) = 1/m * X(:, 1)' * (sigmoid(X * theta) - y);
grad(2:end) = 1/m * X(:, 2:end)' * (sigmoid(X * theta) - y) + lambda / m * theta(2:end);

end
