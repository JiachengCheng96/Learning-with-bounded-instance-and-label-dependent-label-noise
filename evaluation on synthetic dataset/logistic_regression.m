function [f, g] = logistic_regression(theta, X, y)  
    %  
    % Arguments:  
    %   theta - A column vector containing the parameter values to optimize.  
    %   X - The examples stored in a matrix.    
    %       X(i,j) is the i'th coordinate of the j'th example.  
    %   y - The label for each example.  y(j) is the j'th example's label.  
    %  

    % initialize objective value and gradient.  
    f = 0;  
    g = zeros(size(theta));  
    
  
    % store the objective function value in 'f', and the gradient in 'g'.  
    f = -sum(y.*log(sigmoid(theta'*X)) + (1-y).*log(1 - sigmoid(theta'*X)));  
    g = X*(sigmoid(theta'*X) - y)';  
end