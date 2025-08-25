function [tr, est] = xtrace_reiso(B,n,s)
% Input:  Function B() computing matrix products B(X) = B*X, number
%         of rows n, and number of matvecs s
% Output: Estimate tr of trace(B), estimate est of the error
%         abs(tr - trace(B))

% Define test matrix 
k = floor(s/2);         % Approximation rank is s/2
Om = randn(n,k);        % Gaussian matrix

% Randomized SVD and downdate
Y = B(Om);              % Collect matvecs
[Q,R] = qr(Y,"econ");   % Randomized SVD
S = cnormc(inv(R'));    % Downdate RSVD, cnormc normalizes columns

% Compute other necessary matrices
Z = B(Q);               % Collect matvecs
H = Q'*Z;
W = Q'*Om;
T = Z'*Om;
X = W - S .* diagprod(W,S).';

% Scaling factor
alpha = (n - k + 1) ./ (sqcolnorms(Om) - sqcolnorms(X));

% Compute estimator, output
tr_vec = trace(H) * ones(k,1) - diagprod(S,H*S)...
     + alpha .* (-diagprod(T,X) + diagprod(X,H*X)...
                 + diagprod(W, S) .* diagprod(S, R));
tr = mean(tr_vec);           % Trace estimate
est = std(tr_vec) / sqrt(k); % Error estimate

end