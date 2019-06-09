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
%x = [ones(size(X, 1), 1) X];
actual_cost_v = y;

prediction_cost_v = X*theta;
cost_diff_v = prediction_cost_v.- actual_cost_v;
cost_diff_sqr = (cost_diff_v'*cost_diff_v);

cost =  cost_diff_sqr/(2*m);


theta1 = theta;
theta1(1) = 0;

regularization =  (theta1'*theta1) * (lambda/(2*m));

J = cost + regularization;


cost_x_X = cost_diff_v.* X;

cost_x_X = (cost_x_X' * ones(m,1));

cost_x_X = cost_x_X/m;


theta2 = theta1 *(lambda/m);

grad = cost_x_X.+ theta2;











% =========================================================================

grad = grad(:);

end
