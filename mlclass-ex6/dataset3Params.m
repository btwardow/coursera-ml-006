function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

C_eval = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];
sigma_eval = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];
min_error_rate = 1;

for _C = C_eval
    for _sigma = sigma_eval
        fprintf('\nTrening SVN with C=%f and sigma=%f', _C, _sigma); 
        model= svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma));
        predictions = svmPredict(model, Xval);
        error_rate = mean(double(predictions ~= yval));
        fprintf('\nMin error rate: %f', min_error_rate);
        fprintf('\nError rate on cross validation set: %f', error_rate);
        if error_rate < min_error_rate
            min_error_rate = error_rate;
            C = _C;
            sigma = _sigma;
            fprintf('\nNew min error rate. Setting C = %f and sigma = %f', C, sigma);
        endif
    end
end

% =========================================================================

end
