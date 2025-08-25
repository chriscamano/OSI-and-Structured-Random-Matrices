clear; clc;

%% Parameters
n         = 2^10;                % ambient dimension
R         = 50;                  % target rank for low-rank + noise
zeta_vals = [1, 2, 3, 4];
ks        = 20:20:300;
num_runs  = 100;

numProfiles = 9;
numZeta     = numel(zeta_vals);
numKs       = numel(ks);

errG_all = zeros(numProfiles, numZeta, numKs, num_runs);
errS_all = zeros(numProfiles, numZeta, numKs, num_runs);
errI_all = zeros(numProfiles, numZeta, numKs, num_runs);  

%% Pre-build all nine test matrices 
As = cell(9,1);

% 1–3: Low‐rank + noise
xi_vals = [1e-4, 1e-2, 1e-1];
for i = 1:3
    G      = randn(n,n);
    E      = G * G';
    D0     = diag([ones(R,1); zeros(n-R,1)]);
    As{i}  = D0 + (xi_vals(i)/n) * E;
end

% 4–6: Polynomial decay (diagonal)
p_vals = [0.5, 1, 2];
for i = 1:3
    v            = zeros(n,1);
    v(1:R)       = 1;
    tail         = (2:(n-R+1)).^(-p_vals(i));
    v(R+1:end)   = tail';
    As{3+i}      = diag(v);
end

% 7–9: Exponential decay (diagonal)
q_vals = [0.01, 0.1, 0.5];
for i = 1:3
    v            = zeros(n,1);
    v(1:R)       = 1;
    tail         = 10 .^ (-(1:(n-R))' * q_vals(i));
    v(R+1:end)   = tail;
    As{6+i}      = diag(v);
end

%% Profile names
profile_names = { ...
  'LowRankLowNoise','LowRankMedNoise','LowRankHiNoise', ...
  'PolyDecaySlow','PolyDecayMed','PolyDecayFast', ...
  'ExpDecaySlow','ExpDecayMed','ExpDecayFast'};

%% Main loops
totalTasks = numProfiles * numZeta * numKs;
task = 0;

for p = 1:numProfiles
    A_full    = As{p};
    normA_fro = norm(A_full, 'fro');

    for zi = 1:numZeta
        zeta = zeta_vals(zi);
        for ki = 1:numKs
            k = ks(ki);
            task = task + 1;
            fprintf('Task %d/%d: %s, ζ=%d, k=%d\n', ...
                    task, totalTasks, profile_names{p}, zeta, k);

            errG = zeros(num_runs,1);
            errS = zeros(num_runs,1);
            errI = zeros(num_runs,1);

            for run = 1:num_runs
                % Gaussian sketch
                OmegaG    = randn(n, k);
                YG        = A_full * OmegaG;
                QG        = orth(YG);
                errG(run) = compute_error_mat(A_full, QG, normA_fro);

                % sparse-sign i-subcols sketch
                OmegaS    = sparseStack(k, n, zeta)';
                YS        = A_full * OmegaS;
                QS        = orth(YS);
                errS(run) = compute_error_mat(A_full, QS, normA_fro);

                % iid-column sketch
                OmegaI    = sparse_iid(n, k, zeta);
                YI        = A_full * OmegaI;
                QI        = orth(YI);
                errI(run) = compute_error_mat(A_full, QI, normA_fro);
            end

            errG_all(p,zi,ki,:) = errG;
            errS_all(p,zi,ki,:) = errS;
            errI_all(p,zi,ki,:) = errI;
        end
    end
end

%% Save results
save('zeta_testsSS_fullA_updatedRSVD_iid.mat', ...
     'ks','zeta_vals','profile_names', ...
     'errG_all','errS_all','errI_all');
fprintf('Saved to %s\n', fullfile(pwd, 'zeta_testsSS_fullA_updatedRSVD_iid.mat'));

function err = compute_error_mat(A, Q, normA)
    B   = Q' * A;
    [U,S,V] = svd(B, 'econ');
    A_k = Q * U * S * V';
    err = norm(A - A_k, 'fro') / normA;
end

function S = sparse_iid(d, k, zeta)
    p = zeta / k;
    [rows, cols] = find(rand(d, k) < p);
    signs = 2*randi([0,1], numel(rows), 1) - 1;
    S = sqrt(1/zeta) * sparse(rows, cols, signs, d, k);
end
