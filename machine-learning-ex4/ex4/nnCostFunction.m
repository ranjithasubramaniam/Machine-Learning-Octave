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
term3=zeros(m,1);
reg1=0;
reg2=0;
grad2_acc=zeros(size(Theta2));
grad1_acc=zeros(size(Theta1));
for i=1:m,

a1=[1 X(i,:)];
z2= Theta1*a1';
a2= [1 sigmoid(z2)'];
z3=Theta2*a2';
a3=sigmoid(z3);
hx=a3;
newy=zeros(num_labels,1);
newy(y(i),1)=1;
term1 = newy.*log(hx);
term2 = (1-newy).*log(1-hx);
term3(i,1)=sum(-term1-term2);


del3=a3-newy;
del2=(Theta2(:,2:end)'*del3).*(sigmoidGradient(z2));
grad2_acc += del3*a2;
grad1_acc+=del2*a1;
end


for j=1:hidden_layer_size,
 for k=2:(input_layer_size+1),
  
  reg1+=Theta1(j,k).^2;
 
 end
end 

for j=1:num_labels,
 for k=2:(hidden_layer_size+1),
  
  reg2+=Theta2(j,k).^2;
 
 end
end 
J = ((1/m)*sum(term3))+((lambda/(2*m))*(reg1+reg2));

Theta1_grad(:,1) = (1/m).*grad1_acc(:,1);
Theta2_grad(:,1) = (1/m).*grad2_acc(:,1);

for i=1:hidden_layer_size,
 for j=2:(input_layer_size+1),
  
  Theta1_grad(i,j) = (1/m).*grad1_acc(i,j)+(lambda/m)*Theta1(i,j);
 
 end
end 

for i=1:num_labels,
 for j=2:(hidden_layer_size+1),
  
  Theta2_grad(i,j) = (1/m).*grad2_acc(i,j)+(lambda/m)*Theta2(i,j);
 
 end
end 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
