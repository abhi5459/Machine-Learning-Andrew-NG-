function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
X=[ones(m,1) X];
num_labels1=size(Theta1,1);
num_labels2 = size(Theta2, 1);

A=sigmoid(X*Theta1');
A=[ones(m,1) A];


% You need to return the following variables correctly 
p = zeros(m, 1);

[c,p]=max(sigmoid(A*Theta2'),[],2);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
