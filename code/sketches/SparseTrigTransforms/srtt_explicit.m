% Copyright (c) 2025, Chris Camano
function S = srtt_explicit(d, k, rounds, transform, sketch_type, gamma)
% SRRT_EXPLICIT   Build S = sqrt(d/k) * (D * H * S0)
%                where S0 is your sparse selector and H ∈ {dct,fwht,fourier}.
  if nargin < 6, gamma       = [];         end
  if nargin < 5, sketch_type = 'uniform';  end
  if nargin < 4, transform   = 'dct';      end
  if nargin < 3, rounds      = 1;          end

  switch lower(sketch_type)
    case 'uniform'
      %idx = randperm(d, k,1);                  % Without replacement
      idx = randi(d, 1, k);                     % With replacement
      S0  = sparse(idx, 1:k, 1, d, k) ;         

    case 'iid'
      p = gamma/d;                              
      [rows,cols] = find(rand(d,k) < p);            
      V     = 2*randi([0,1], numel(rows), 1) - 1;  
      S0    = sparse(rows, cols, V, d, k);

    case 'countsketch'
     cols  = randi(k, d, 1);                   
     signs = 2*randi([0,1], d, 1) - 1;         
     S0    = sparse((1:d)', cols, signs, d, k);

    case 'sparsecol'
      if isempty(gamma), error('SparseCol needs γ'); end
      nnz_tot = k * gamma;
      rows = zeros(1, nnz_tot);
      cols = zeros(1, nnz_tot);
      V = zeros(1, nnz_tot);
      ptr = 1;
      for j = 1:k
        idx = randperm(d, gamma);                 % γ entries in column j
        sgns = randi([0,1], 1, gamma)*2 - 1;      % Rademachers
        rows(ptr:ptr+gamma-1) = idx;
        cols(ptr:ptr+gamma-1) = j;
        V(ptr:ptr+gamma-1) = sgns;
        ptr = ptr + gamma;
      end
      S0 = sparse(rows, cols, V, d, k);

    otherwise
      error('Unknown sketch_type "%s"', sketch_type);
  end

  Dsign = randi([0,1], d, rounds)*2 - 1;
  M = S0;
  for t = 1:rounds
    switch lower(transform)
      case 'dct'
        M = dct(full(M), [], 1);
      case 'fwht'
        M = fwht_mex(full(M));
      case 'fourier'
        M = fft(full(M), [], 1) / sqrt(d);
      otherwise
        error('Unknown transform "%s".', transform);
    end
    M = bsxfun(@times, Dsign(:,t), M);
  end
  S = sqrt(d/k) * M;
end
