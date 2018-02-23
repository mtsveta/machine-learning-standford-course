function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Make predictions using learned neural network. 
% Set p to a vector containing labels between 1 to num_labels.

%{
a1 = X; % [5000 x 401]

a1 = [ones(m, 1), a1]; % 5000 x 401
z2 = a1 * Theta1'; % [5000 x 401] x [401 x 25] = [5000 x 25]
a2 = sigmoid(z2); % [5000 x 25]

a2 = [ones(m, 1), a2]; % [5000 x 26]
z3 = a2 * Theta2'; % [5000 x 26] x [26 x 10] = [5000 x 10]
a3 = sigmoid(z3); % [5000 x 10]

h_theta = a3;
%}

a = X;
Theta = {Theta1, Theta2};   % we need to store the sequence of the Thetas to fetch them in the loop
for i = 1 : 2
    a = [ones(m, 1), a];    % extending a by column of ones: obtaining m x s_i, s_i = s_{i-1} + 1
    z = a * Theta{i}';      % [m x s_i] x [s_i x s_{i+1}] = [m x s_{i+1}]
    a = sigmoid(z);         % [m x s_{i+1}]
end
h_theta = a;

[~, p] = max(h_theta, [], 2);

% =========================================================================

end
