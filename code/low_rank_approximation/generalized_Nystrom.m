% Copyright (c) 2025, Chris Camano

function [Ahat, elapsed, relErr] = generalized_Nystrom(A, Om, Psi, recordTime)
% GENERALIZED_NYSTROM (stabilized) 
% This version of Generalized Nystrom follows Yuji nakatsukasa's
% "Fast and stable randomized low-rank matrix approximation (2020)" See
% Sec. 5.1 For additional details
%   Inputs:
%     A          — (n×d) data matrix
%     Om         — (d×k) right sketch
%     Psi        — (n×l) left sketch (l ≥ k)
%     recordTime — logical (optional)
%   Outputs:
%     Ahat   — rank‐k approximation of A
%     elapsed — time to form sketch + QR + triangular solve
%     relErr — ‖A − Ahat‖_F/‖A‖_F (if recordTime)

  if nargin < 4, recordTime = false; end
  if recordTime, t0 = tic; end

  % Sketches
  AX = A * Om;            % (n × k)
  YA = Psi' * A;          % (l × d)
  [Q, R] = qr(Psi' * AX, 0);    % QR of the small core

  % stable pseudoinverse
  U = AX / R;             % (n×k)
  V = (Q' * YA);          % (d×k)

  if recordTime
    elapsed = toc(t0);
    Ahat = U * V;         
    relErr  = norm(A - Ahat, 'fro') / norm(A, 'fro');
  else
    elapsed = [];
    relErr  = [];
  end
end
