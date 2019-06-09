function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

possibleValue = {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30};
C = 1;
sigma = 0.3;
prv_error =100000;
cur_error=0;
temp_c = 0.0;
temp_sigma = 0.0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
for i=1:length(possibleValue)
  temp_c = possibleValue{i};
  for j=1:length(possibleValue)
    temp_sigma = possibleValue{j};
    %printf("temp_sigma %f , temp_c %f \n", temp_sigma, temp_c);
    model= svmTrain(X, y, temp_c, @(Xval, yval) gaussianKernel(Xval, yval, temp_sigma));
    %printf("model predicted by temp values %o \n" , model);
    predictions = svmPredict(model, Xval);
    cur_error = mean(double(predictions ~= yval));
    %printf("current error %f \n", cur_error);
    if(cur_error < prv_error)
      %printf("cur_error %f is small from pre_error %f \n", cur_error, prv_error);
      C = temp_c;
      sigma=temp_sigma;
      prv_error = cur_error;
    endif
  endfor
endfor







% =========================================================================

end
