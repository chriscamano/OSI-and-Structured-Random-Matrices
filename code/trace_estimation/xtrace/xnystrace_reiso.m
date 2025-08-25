function [tr,est] = xnystrace_reiso(A,n,s)
% Input:  Function A() computing matrix products A(X) = A*X, number
%         of rows n, and number of matvecs s
% Output: Estimate tr of trace(A), estimate est of the error
%         abs(tr - trace(A))

Om = randn(n,s);                % Gaussian random test matrix
Y = A(Om);                      % Matrix product Y = A*Om
mu = eps*norm(Y,"fro")/sqrt(n); % Compute shift
Y = Y + mu * Om;                % Apply shift to Y
H = Om'*Y;
R = chol((H+H')/2);             % Explicitly symmetrize H to be safe
F = Y/R;                        % Triangular substitution

% Downdating
Z = (F/R') .* (sqrownorms(inv(R)) .^ (-1/2))';

% Reisotropization
T = chol(Om'*Om);
Tinv = inv(T);
X = T - Tinv' / diag(sqcolnorms(Tinv'));
alpha = (n-s+1) ./ (sqcolnorms(Om) - sqcolnorms(X)); 

% Compute vector of estimates
tr_vec = norm(F,"fro")^2 * ones(s,1) - sqcolnorms(Z) ...
    + alpha .* abs(diagprod(Z,Om)) .^ 2 - mu * n * ones(s,1);
tr = mean(tr_vec);           % Trace estimate
est = std(tr_vec) / sqrt(s); % Error estimate

end

