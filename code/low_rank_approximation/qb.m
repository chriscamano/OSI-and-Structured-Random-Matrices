% Copyright (c) 2025, Chris Camano

function [Q, B, elapsed, relErr] = qb(A, Omega, recordTime)
% QB   Compute randomized QB approximation .
%   [Q,B,elapsed,relErr] = QB(A,Omega,recordTime)
%
%   Inputs:
%     A          — (n×m) data matrix
%     Omega      — (m×k) sketching matrix
%     recordTime — logical flag; if true, measure time and compute error
%
%   Outputs:
%     Q (n×k), B (k×m) so that A ≈ Q*B
%     elapsed    — time for sketch/QR steps (empty if recordTime=false)
%     relErr     — ‖A − Q*B‖_F / ‖A‖_F (empty if recordTime=false)

  if nargin<3, recordTime = false; end
  if recordTime, t0 = tic; end

  % form sketch and orthonormal basis
  Y = A * Omega;           
  [Q,~] = qr(Y,0);
  B = Q' * A;              

  if recordTime
    elapsed = toc(t0);
    Arec    = Q * B;
    relErr  = norm(A - Arec, 'fro') / norm(A, 'fro');
  else
    elapsed = [];
    relErr  = [];
  end
end
