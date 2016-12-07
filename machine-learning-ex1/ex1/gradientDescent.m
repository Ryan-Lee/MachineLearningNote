function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    sum0 = 0;
    sum1 = 0;
    for i = 1:m
        sum0 = sum0 + (theta' * X(i,:)' - y(i,1));
        sum1 = sum1 + (theta' * X(i,:)' - y(i,1)) * X(i,2);     
    end;
    theta0 = theta(1,1) - alpha / m * sum0;
    theta1 = theta(2,1) - alpha / m * sum1;
    theta = [theta0; theta1];

    %printf('theta:%f,%f\n',theta(1,1),theta(2,1));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %printf('J(theta):%f\n',J_history(iter,1));

end

end
