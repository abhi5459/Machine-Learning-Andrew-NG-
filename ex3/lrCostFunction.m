function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); 
grad = zeros(length(theta),1);
h=sigmoid(X*theta);
tsquare = lambda/(2*m) * (sum(theta .* theta) - theta(1).*theta(1));
J=(-1/m *( y' *log(h) + (1-y)' * log(1-h)) + tsquare);

s=X(:,1);
grad(1)= 1/m * sum(s .* (h-y));
for i=2:length(theta)
    s=X(:,i);
    grad(i)= 1/m * (sum(s .* (h-y)) + lambda .* theta(i));
end

end
