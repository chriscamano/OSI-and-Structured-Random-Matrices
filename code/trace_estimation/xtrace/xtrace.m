function [tr, est] = xtrace(B, n, s, mode, varargin)
% xtrace   Trace estimation via X-trace with flexible sketch modes
%
%   [tr, est] = xtrace(B, n, s)
%   [tr, est] = xtrace(B, n, s, mode)
%   [tr, est] = xtrace(B, n, s, mode, l, dist, sphere)
%
% Inputs
%   B      — function handle B(X) = B*X
%   n      — number of rows/cols of B
%   s      — total number of matvecs
%   mode   — 'gaussian' (default), 'signs', or 'kr'
%   l      — (KR only) number of factors in Khatri–Rao (default 2)
%   dist   — (KR only) 'gaussian' (default), 'cgaussian', 'rademacher', 'crademacher'
%   sphere — (KR only) normalize each column to unit ℓ₂-norm (default true)
%
% Outputs
%   tr   — estimated trace
%   est  — approximate standard error (abs(tr - trace(B)) ≈ est)

  % compute half of s for the sketch dimension
  k = floor(s/2);

  % default mode if missing or empty
  if nargin < 4 || isempty(mode)
    mode = 'gaussian';
  end

  switch lower(mode)
    case 'gaussian'
      Om = randn(n, k);
      Om = Om./vecnorm(Om);
      Om = Om*sqrt(n);

    case 'signs'
      Om = random_signs(n, k);

    case 'kr'
      % defaults for KR sketch
      l      = 2;
      dist   = 'cgaussian';
      sphere = true;
      % override if provided in varargin
      if numel(varargin) >= 1 && ~isempty(varargin{1})
        l = varargin{1};
      end
      if numel(varargin) >= 2 && ~isempty(varargin{2})
        dist = varargin{2};
      end
      if numel(varargin) >= 3 && ~isempty(varargin{3})
        sphere = varargin{3};
      end
      Om = krSketch(n, k, l, dist, sphere);

    otherwise
      error('xtrace:unknownMode', 'Unrecognized mode "%s".', mode);
  end

  % randomized SVD and downdate
  Y      = B(Om);
  [Q, R] = qr(Y, 'econ');
  S      = cnormc(inv(R'));

  % additional matvecs
  Z = B(Q);
  H = Q' * Z;
  W = Q' * Om;
  T = Z' * Om;
  X = W - S .* diagprod(W, S).';

  % per-column trace estimates
  tr_vec = trace(H) * ones(k,1) ...
         - diagprod(S, H*S) ...
         - diagprod(T, X) ...
         + diagprod(X, H*X) ...
         + diagprod(W, S) .* diagprod(S, R);

  tr  = mean(tr_vec);
  est = std(tr_vec) / sqrt(k);
end