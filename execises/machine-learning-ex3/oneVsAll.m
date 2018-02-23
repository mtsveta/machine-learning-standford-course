function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

    % Some useful variables
    m = size(X, 1);
    n = size(X, 2);

    % You need to return the following variables correctly 
    all_theta = zeros(num_labels, n + 1);
    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    for i = 1 : num_labels

        % Define initial theta
        initial_theta = zeros(n+1, 1);

        % Defined cost functional
        [~, ~] = lrCostFunction(initial_theta, X, y, lambda);
        options = optimset('GradObj', 'on', 'MaxIter', 400);

        % Optimize the costs
        % (y == i) so that everyone who is 
        % equal to i    -> is treated as 1
        % nonequal to i -> is treated as 0
        [theta, ~, ~] = fmincg(@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
                               initial_theta, options);
                           
        % Write the obtained result into theta 
        all_theta(i, :) = theta';
    end

end
