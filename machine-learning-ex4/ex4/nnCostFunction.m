function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_e = [];
for k = 1:num_labels
  y_e = [y_e;k];
endfor
cost_m = 0.0;
for i=1:m
   %Calculating output using nueral network for ith example
   x_i = X(i,:);
   x_i = [1,x_i];
   thetaTxi = (Theta1)*(x_i');
   log_h_xi = sigmoid(thetaTxi);

   log_h_xi = [1;log_h_xi];

   thetaTx2 = (Theta2)*(log_h_xi);
   y_i = sigmoid(thetaTx2);

  %Calculating cost for ith example output from nueral network(above)
  t = y(i,:);
  output_y = (y_e == t);

  log_h_x = log(y_i);
  x1 = -((output_y)'*log_h_x);

  one_minus_log_h_X = log(1-y_i);
  one_minus_y = (1-output_y);
  x2 = one_minus_y'* one_minus_log_h_X;

  %Calculation cost for
  cost_k = (x1-x2);

  %Adding the cost
  cost_m = cost_m + cost_k;

endfor

% Calculating grad using backpropagation

DELTA1 = 0;
DELTA2 = 0;

for i=1:m

  %step 1
  a1 = X(i,:);
  a1 = [1,a1];

  z2 = (Theta1) * (a1');
  a2 = sigmoid(z2);

  a2 = [1;a2];
  z3 = (Theta2)* (a2);
  a3 = sigmoid(z3);

  t= y(i,:);
  output_y = (y_e==t);

  %step 2
  delta3 = (a3 - output_y);

  %step 3
  delta2 = ((Theta2(:,2:end))'*delta3).*sigmoidGradient(z2);

  %delta2 = delta2(2:end);

  %step 4
  DELTA2 = DELTA2 + delta3*(a2)';

  DELTA1 = DELTA1 + delta2 * (a1);

endfor

if lambda == 0
  J = cost_m/m;

  DELTA1_0 = (1/m).*DELTA1;
  DELTA2_0 = (1/m).*DELTA2;

  grad = [DELTA1_0(:);DELTA2_0(:)];

elseif lambda != 0
   % Regularized cost function
    theta1_vector_without_first_col = nn_params((1+hidden_layer_size):(hidden_layer_size*(input_layer_size +1)));
    theta2_vector_without_first_col = nn_params((1 + num_labels + (hidden_layer_size * (input_layer_size + 1))) :end);

    r1 = theta1_vector_without_first_col'*theta1_vector_without_first_col;

    r2 = theta2_vector_without_first_col'*theta2_vector_without_first_col;
    regularization = ((r1+r2) * lambda)/(2*m);
    J = (cost_m/m) + regularization;

    %Regularized gradients
    regularization1 = (lambda/m).*theta1_vector_without_first_col;

    DELTA1 = (1/m).*DELTA1;

    for n =1:size(regularization1)
      DELTA1(n+hidden_layer_size) = DELTA1(n + hidden_layer_size) + regularization1(n);
    endfor


    regularization2 = (lambda/m).*theta2_vector_without_first_col;
    DELTA2 = (1/m).*DELTA2;
    for n =1:size(regularization2)
      DELTA2(n+num_labels) = DELTA2(n + num_labels) + regularization2(n);
    endfor

    grad = [DELTA1(:);DELTA2(:)];

endif

% -------------------------------------------------------------

% =========================================================================

end
