function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %





    %sum0 = 0;
    %sum1 = 0;
    n = length(theta);
    sum = zeros(n,1);
    for i = 1:m
        %sum0 = sum0 + (theta' * X(i,:)' - y(i,1));
        %sum1 = sum1 + (theta' * X(i,:)' - y(i,1)) * X(i,2);
        for j = 1:n
            sum(j,1) = sum(j,1) + (X(i,:) * theta - y(i,1)) * X(i,j);
        end;
    end;
    %theta0 = theta(1,1) - alpha / m * sum0;
    %theta1 = theta(2,1) - alpha / m * sum1;
    for j = 1:n
        theta(j,1) = theta(j,1) - alpha / m * sum(j,1);
    end;
    %theta = [theta0; theta1];





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
