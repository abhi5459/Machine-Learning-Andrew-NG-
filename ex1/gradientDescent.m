function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
c=size(X,2);
for iter = 1:num_iters
    A=X*theta -y;
    f=zeros(c,1);
    for i=1:c
        s=A.*X(:,i);
        f(i) = theta(i) - alpha*(1/m)*sum(s);
    end
    for i=1:c
        theta(i)=f(i);
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
