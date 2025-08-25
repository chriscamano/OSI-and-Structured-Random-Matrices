function [F,mu,Om,R] = nystrom(A,n,k)
% Input:  Function A() computing matrix products A(X) = A*X, number 
%         of rows n, rank k
% Output: Shifted Nystrom approximation Ahat = F*F', represented by 
%         factor F, shift mu, test matrix Om

Om = random_signs(n,k);         % Test matrix of random signs
Y = A(Om);                      % Matrix product Y = A*Om
mu = eps*norm(Y,"fro")/sqrt(n); % Compute shift
Y = Y + mu * Om;                % Apply shift to Y
H = Om'*Y;
R = chol((H+H')/2);             % Explicitly symmetrize H to be safe
F = Y/R;                        % Triangular substitution

end