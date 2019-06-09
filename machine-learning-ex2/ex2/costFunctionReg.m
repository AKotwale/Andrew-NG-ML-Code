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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



thetaTx = X*theta;
log_h_x = log(sigmoid(thetaTx));

x1 = -((y)' * log_h_x);

one_minus_log_h_x = log(1-sigmoid(thetaTx));
one_minus_y = (1-y);

x2 = one_minus_y'* one_minus_log_h_x;

temp = (x1-x2)/m;

theta_without_first_index = theta;
theta_without_first_index(1:1) = 0;



temp1 = (theta_without_first_index'* theta_without_first_index) * lambda;
J = temp + (temp1/(2*m));


const = lambda/m;
theta(1:1) = 0;

theta = theta.*const;

grad1 = (sigmoid(thetaTx).-y);
grad = (X'*grad1)/m;

grad = grad + theta;

% =============================================================

end
