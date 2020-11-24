function [J, grad] = costFunction(theta, X, y)

m = length(y); 
grad = zeros(length(theta),1);
h=sigmoid(X*theta);
J=(-1/m *( y' *log(h) + (1-y)' * log(1-h)));

for i=1:length(theta)
    s=X(:,i);
    grad(i)= 1/m*(sum(s .* (h-y)));
end
    
end
