%% zeta_testsSS_varyZeta_fullA.m
% Compare Gaussian vs. uniform, countsketch (ζ=1),
% sparsecol (ζ=1:4), and iid Bernoulli–±1 (ζ=1:4)
% on nine full test matrices.

clear; clc;
nc = feature('numcores');  

% Then tell MATLAB to try and use them all for multithreaded ops
maxNumCompThreads(nc);
%% — Parameters —
n          = 2^10;               % ambient dimension
R          = 50;                 % target rank + noise cutoff
zeta_vals  = 1:4;                % sparsity parameters for sparsecol & iid
ks         = 20:20:300;          % sketch dimensions to sweep
num_runs   = 100;                  % independent trials

sketch_types = {'gaussian','uniform','countsketch','sparsecol','iid'};
numSketch    = numel(sketch_types);
numZeta      = numel(zeta_vals);

%% — Preallocate error storage —
numProfiles = 9;
numKs       = numel(ks);
err_all     = zeros(numProfiles, numKs, numSketch, numZeta, num_runs);

%% — Build the nine full matrices A —
As = cell(numProfiles,1);
profile_names = { ...
  'LowRankLowNoise','LowRankMedNoise','LowRankHiNoise', ...
  'PolyDecaySlow','PolyDecayMed','PolyDecayFast', ...
  'ExpDecaySlow','ExpDecayMed','ExpDecayFast'};

% 1–3: Low‐rank + noise
xi_vals = [1e-4, 1e-2, 1e-1];
for i = 1:3
  G   = randn(n,n);
  E   = G * G';                     
  D0  = diag([ones(R,1); zeros(n-R,1)]);
  As{i} = D0 + (xi_vals(i)/n) * E;
end

% 4–6: Polynomial decay (diagonal embedded in full A)
p_vals = [0.5, 1, 2];
for i = 1:3
  v = zeros(n,1);
  v(1:R) = 1;
  tail = (2:(n-R+1)).^(-p_vals(i));
  v(R+1:end) = tail';
  As{3+i} = diag(v);
end

% 7–9: Exponential decay (diagonal embedded in full A)
q_vals = [0.01, 0.1, 0.5];
for i = 1:3
  v = zeros(n,1);
  v(1:R) = 1;
  tail = 10 .^ (-(1:(n-R))' * q_vals(i));
  v(R+1:end) = tail;
  As{6+i} = diag(v);
end

%% — Main loops: profile × k × sketch × ζ(if sparsecol/iid) × runs —
for p = 1:numProfiles
  A       = As{p};
  normA_f = norm(A,'fro');

  for ki = 1:numKs
    k = ks(ki);

    for si = 1:numSketch
      sketch = sketch_types{si};

      if strcmp(sketch,'sparsecol') || strcmp(sketch,'iid')
        for zi = 1:numZeta
          zeta = zeta_vals(zi);
          for run = 1:num_runs
            fprintf('p=%d, k=%d, %s ζ=%d, run=%d/%d\n', ...
                    p, k, sketch, zeta, run, num_runs);
            Omega = srtt_explicit(n, k, 1, 'dct', sketch, zeta);
            err_all(p,ki,si,zi,run) = ...
              compute_error_fullA(A, Omega, normA_f);
          end
        end

      else
        zi = 1; zeta = 1;
        for run = 1:num_runs
          fprintf('p=%d, k=%d, %s ζ=%d, run=%d/%d\n', ...
                  p, k, sketch, zeta, run, num_runs);
          switch sketch
            case 'gaussian'
              Omega = randn(n, k);
            otherwise  % 'uniform' or 'countsketch'
              Omega = srtt_explicit(n, k, 1, 'dct', sketch, zeta);
          end

          err_all(p,ki,si,zi,run) = ...
            compute_error_fullA(A, Omega, normA_f);
        end
      end

    end
  end
end

%% — Save results —
save('stt_sparsity_LRA_varyZeta_fullA.mat', ...
     'ks','zeta_vals','profile_names','sketch_types','err_all');
fprintf('Saved results to %s\n', fullfile(pwd,'stt_sparsity_LRA_varyZeta_fullA.mat'));

%% — Helper: compute error on full A via rSVD projection —
function err = compute_error_fullA(A, Omega, normA_f)
  Y = A * Omega;
  % [Q,~] = qr(Y,0);
  Q =  orth(Y);
  B     = Q' * A;
  [U,S,V] = svd(B,'econ');
  Ak    = Q * U * S * V';
  err   = norm(A - Ak,'fro') / normA_f;
end