function [tr, est] = xnystrace(A, n, s, varargin)
% xnystrace  Trace estimation via X-Nyström with three sketch modes
%
%   [tr, est] = xnystrace(A, n, s)
%   [tr, est] = xnystrace(A, n, s, mode)
%   [tr, est] = xnystrace(A, n, s, mode, l, dist, sphere)
%
% Inputs
%   A      — function handle A(X) = A*X
%   n      — number of rows/cols of A
%   s      — total number of mat-vec products
%   mode   — 'gaussian' (default), 'signs', or 'kr'
%   l      — (KR only) number of factors (default 2)
%   dist   — (KR only) distribution (default 'gaussian')
%   sphere — (KR only) normalize columns (default true)
%
% Outputs
%   tr   — estimated trace
%   est  — standard-error estimate

  % parse mode
  if numel(varargin) >= 1 && ~isempty(varargin{1})
    mode = lower(varargin{1});
  else
    mode = 'gaussian';
  end

  switch mode
    case 'gaussian'
      Om = randn(n, s);
      Om = Om./vecnorm(Om);
      Om = Om*sqrt(n);


    case 'signs'
      Om = random_signs(n, s);

    case 'kr'
      % defaults for KR
      l      = 2;
      dist   = 'cgaussian';
      sphere = true;
      % override from varargin
      if numel(varargin) >= 2 && ~isempty(varargin{2})
        l = varargin{2};
      end
      if numel(varargin) >= 3 && ~isempty(varargin{3})
        dist = varargin{3};
      end
      if numel(varargin) >= 4 && ~isempty(varargin{4})
        sphere = varargin{4};
      end
      Om = krSketch(n, s, l, dist, sphere);

    otherwise
      error('xnystrace:unknownMode', 'Unrecognized mode ''%s''.', mode);
  end

  Y  = A(Om);
  mu = eps * norm(Y, 'fro') / sqrt(n);
  Y  = Y + mu * Om;

  H  = Om' * Y;
  warning('off','MATLAB:posdef');        % suppress near‐singular warnings
  R  = chol((H + H')/2);
  F  = Y / R;

  % downdate
  Z = (F / R') .* (sqrownorms(inv(R)).^(-1/2))';

  % compute per-column trace contributions
  tr_vec = norm(F, 'fro')^2 * ones(s,1) ...
         - sqcolnorms(Z) ...
         + abs(diagprod(Z, Om)).^2 ...
         - mu * n * ones(s,1);

  tr  = mean(tr_vec);
  est = std(tr_vec) / sqrt(s);
end