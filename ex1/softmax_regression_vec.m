function [f,g] = softmax_regression_vec(theta, X, y)
%
% Arguments:
%   theta - A vector containing the parameter values to optimize.
%       In minFunc, theta is reshaped to a long vector.  So we need to
%       resize it to an n-by-(num_classes-1) matrix.
%       Recall that we assume theta(:,num_classes) = 0.
%
%   X - The examples stored in a matrix.
%       X(i,j) is the i'th coordinate of the j'th example.
%   y - The label for each example.  y(j) is the j'th example's label.
%
m=size(X,2);
n=size(X,1);

% theta is a vector;  need to reshape to n x num_classes.
theta=reshape(theta, n, []);
num_classes=size(theta,2)+1;

% initialize objective value and gradient.
f = 0;
g = zeros(size(theta));

%
% TODO:  Compute the softmax objective function and gradient using vectorized code.
%        Store the objective function value in 'f', and the gradient in 'g'.
%        Before returning g, make sure you form it back into a vector with g=g(:);
%
%%% YOUR CODE HERE %%%

thetaz = [theta, zeros(n, 1)];
a = thetaz.' * X;
% Prevent overflow
a = bsxfun(@minus, a, max(a, [], 1));
P = exp(a);
P = bsxfun(@rdivide, P, sum(P));
ind = sub2ind(size(P), y, 1 : size(P, 2));
f = -sum(log(P(ind)));

P(ind) = P(ind) - 1;
g = X * P(1 : end - 1, :).';

% f2 = 0;
% for iSamp = 1 : m
%     for kClass = 1 : num_classes
%         P = exp(thetaz(:, kClass).' * X(:, iSamp)) / sum(exp(thetaz.' * X(:, iSamp)));
%         if y(iSamp) == kClass
%             f2 = f2 - P;
%         end
%     end % for kClass
% end % for iSamp
% 
% g2 = zeros(size(thetaz));
% for iSamp = 1 : m
%     for kClass = 1 : num_classes
%         P2 = exp(thetaz(:, kClass).' * X(:, iSamp)) / sum(exp(thetaz.' * X(:, iSamp)));
%         for coord = 1 : n
%             if y(iSamp) == kClass
%                 g2(coord, kClass) = g2(coord, kClass) + X(coord, iSamp) * (1.0 - P2);
%             else
%                 g2(coord, kClass) = g2(coord, kClass) + X(coord, iSamp) * P2;
%             end
%         end
%     end % for kClass
% end % for iSamp
% 
% f,
% f2,
% g,
% g2

g=g(:); % make gradient a vector for minFunc
