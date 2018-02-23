function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

m = size(X, 1);

% Compute the mean of each feature and subtract it from the dataset
mu = repmat(mean(X), m, 1);

% Compute the standard deviation of each feature and divide the dataset by it 
sigma = repmat(std(X, 0, 1), m, 1);

X_norm = (X - mu) ./ sigma;

end
