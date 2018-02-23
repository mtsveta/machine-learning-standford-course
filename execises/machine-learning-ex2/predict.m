function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples
n = size(X, 2); % Number of freatures 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% Make predictions using learned logistic regression parameters
% p = 1 if sigmoid(X * theta) >= 0.5 and p = 0 otherwise
p = (sigmoid(X * theta) >= 0.5); 

% p = (X * theta >= 0.0); % should be also possible and equivalent to above

% =========================================================================

end
