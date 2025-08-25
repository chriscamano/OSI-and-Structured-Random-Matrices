% Copyright (c) 2025, Chris Camano
function [U, S, V, elapsed, relErr] = generalized_Nystrom_SVD(A, Om, Psi, recordTime)
  if nargin < 4, recordTime = false; end
  if recordTime, t0 = tic; end

  %=========== Sketching ===========
  Y = A * Om;
  [Q,~] = qr(Y,0);       
  X = A' * Psi;
  [P, T] = qr(X, 0); 

  %=========== Stabilized pseduoinverse ===========
 [U1, s1, V1] = svd(Psi' * Q, 'econ', 'vector');
  sigma_max = s1(1);
  tol = 5 * eps(sigma_max);
  r = sum(s1 > tol);

  C = V1(:,1:r) * ((U1(:,1:r)'*T') ./ s1(1:r));

  %=========== SVD of core matrix ===========
  [Uc, S, Vc] = svd(C, 'econ');
  U = Q * Uc;                      
  V = P * Vc;    

  if recordTime
    elapsed = toc(t0);
    Ahat    = U * S * V';
    relErr  = norm(A - Ahat, 'fro') / norm(A, 'fro');
  else
    elapsed = [];
    relErr  = [];
  end
end
