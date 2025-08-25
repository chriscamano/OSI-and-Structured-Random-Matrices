function [t,err,m] = xnystrace_tol(A,N, abstol, reltol, ell, dist, sphere)
% xnystrace_tol  Adaptive X–Nystrom trace estimation with KR test matrices
%
%   [t,err,m] = xnystrace_tol(A, abstol, reltol)
%   [t,err,m] = xnystrace_tol(A, abstol, reltol, l, dist, sphere)
%
% Inputs
%   A       — matrix or matvec handle (via process_matrix)
%   abstol  — absolute error tolerance
%   reltol  — relative error tolerance
%   l       — (optional) number of KR factors (default 2)
%   dist    — (optional) KR distribution (default 'gaussian')
%   sphere  — (optional) normalize KR columns (default true)
%
% Outputs
%   t,err,m — trace estimate, its stderr, and # matvecs used

  [matvec, n] = process_matrix(A, N);
    

  Y   = zeros(n,0);
  Om  = zeros(n,0);
  err = Inf;
  if reltol~=0, t = Inf; else, t = 0; end

  while err >= abstol + reltol * abs(t)
    m_blk = max(2, size(Y,2));

    % Khatri–Rao test matrix
    NewOm   = krSketch(n, m_blk, ell, dist, sphere);
    improved = true;

    Y  = [Y  matvec(NewOm)]; 
    Om = [Om NewOm];          
    [t,err] = xnystrace_helper(Y, Om, improved);
  end

  m = size(Y,2);
end
