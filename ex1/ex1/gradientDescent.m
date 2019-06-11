function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
sum1=0;
sum2=0;

for iter = 1:num_iters
j=X*theta;
for i=1:m
 sum2 = sum2+ ((j(i)-y(i))*X(i));
 sum1=sum1 +(j(i)-y(i));
end;
sum2=sum2/m;
sum1=sum1/m;
temp1=theta(1)-(alpha*sum1);
temp2=theta(2)-(alpha*sum2);
theta1=temp1;
theta2=temp2;
    
    J_history(iter) = computeCost(X, y, theta);
    
    
    end

end
