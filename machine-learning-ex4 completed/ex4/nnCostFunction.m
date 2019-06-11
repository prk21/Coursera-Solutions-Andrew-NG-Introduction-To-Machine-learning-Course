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

x1=[ones(m,1) X];
z1=x1* Theta1';
a1=sigmoid(z1);
a1=[ones(m,1) a1];
z2=a1* Theta2';
a2=sigmoid(z2);
h=a2;

for i=1:num_labels
 c=(y==i);
 J= J+ sum(-c.*log(h(:,i)) -(1-c).*log(1-h(:,i)));
end
J=J/m;
%regularisation
theta1ex=Theta1;
theta1ex(:,1)=0;
theta2ex=Theta2;
theta2ex(:,1)=0;

reg=0.5*lambda/m*(sum(sum(theta1ex.^2)) +sum(sum(theta2ex.^2)));
J=J+reg;

for t=1:m
 a1=X(t,:); %  1x400
 a1=[1 , a1]; %           a1=1x401
 z2=a1*Theta1'; %1x401 401x25 =1x25
 a2=sigmoid(z2); %1x25
 a2=[1,a2]; %           a2=1x26
 z3=a2*Theta2';% 1x26 26x1
 a3=sigmoid(z3);%      a3=1x10
 % a3 is 1x10
 del3=zeros(1,num_labels);% del3= 1x10
 for k=1:num_labels
  del3(k)=a3(k)- (y(t)==k);
  end;
 del2= del3*Theta2.*sigmoidGradient([1,z2]) ;% 1x10 10x26 =1x26 =del2
 del2=del2(2:end);%  del2 =1x25
 Theta1_grad=Theta1_grad + (a1'*del2)';% 401x1 1x25 =401x25
 Theta2_grad=Theta2_grad + (a2'*del3)';%  26x1 1x10 =26x10
 
end;
Theta2_grad=Theta2_grad/m;
Theta1_grad=Theta1_grad/m;

Theta1_grad= Theta1_grad+ lambda/m*theta1ex;
Theta2_grad= Theta2_grad+ lambda/m*theta2ex;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
