function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    
    x0=X(:,1);
    x1=X(:,2);
    h=theta(1)* x0 + (theta(2)* x1);
    
    theta_zero= theta(1) - (alpha/m) * sum((h-y).*x0);
    theta_one = theta(2) - (alpha/m) * sum((h-y).*x1);    
    
    theta= [theta_zero; theta_one];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %disp(J_history(iter));
end

end
