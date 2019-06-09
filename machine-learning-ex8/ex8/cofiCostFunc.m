function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

thetax = (X*Theta');
costdiff = (thetax - Y);
costdiff2 = costdiff.^2;

cost = costdiff2.*R;

J = sum(sum(cost))/2;

reg_theta = sum(sum((Theta.*Theta)));
reg_theta = (reg_theta * lambda)/2;

reg_x = sum(sum((X.*X)));
reg_x = (reg_x * lambda)/2;

J = J + reg_theta + reg_x;

for itr=1:num_movies
  idx = find(R(itr,:) == 1);
  tempTheta = Theta(idx,:);
  tempY = Y(itr,idx);

  X_grad(itr,:) = (X(itr,:) * tempTheta' - tempY) * tempTheta;

  X_grad(itr,:) =  X_grad(itr,:) .+ (lambda*(X(itr,:)));

endfor


for itr=1:num_users
  idx = find(R(:,itr) == 1);
  tempX = X(idx,:);
  tempY = Y(idx,itr);

  Theta_grad(itr,:) = (tempX * Theta(itr,:)' - tempY)' * tempX;

  Theta_grad(itr,:) =    Theta_grad(itr,:) .+(lambda*(Theta(itr,:)));

endfor






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
