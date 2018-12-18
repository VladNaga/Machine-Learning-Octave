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


%% Part 1 implementation

%ones to the data matrix to get a1
A1 = [ones(m, 1) X]; %5000 x 401

%calc first layer
Z1 =  A1 * Theta1'; %5000 x 25
S2 = sigmoid(Z1); %5000 x 25
%ones to the A2 matrix
A2 = [ones(m, 1) S2]; %5000 x 26

%calc output layer
Z2 = A2 * Theta2'; %5000 x 10
A3 = sigmoid(Z2); %5000 x 10

H=A3;

Y = eye(num_labels)(y, :); %5000 x 10

W= -log(H) .* Y - log(1-H) .* (1-Y); %5000 x 10

J= 1/m * sum(sum(W)) + lambda/(2*m) * ... 
        (sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2)));

D3= A3 - Y; %5000 x 10

SG=sigmoidGradient(Z1); %5000 x 25

D2 = (D3 * Theta2(:,2:end)) .* SG; %5000 x 25

W2= D3' * A2;
W1 = D2' * A1;

R1= lambda * Theta1;
R1(:,1)= 0;
R2= lambda * Theta2;
R2(:,1)= 0;

Theta1_grad= (W1 + R1)/m;
Theta2_grad= (W2 + R2)/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
