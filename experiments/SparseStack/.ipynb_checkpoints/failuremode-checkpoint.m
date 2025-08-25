clear; clc; t_start = tic;
%% -- Parameters --
r          = 13;                   % ambient exponent
n          = 2^r;                  % ambient dimension = 1024
trunc_rank = 50;                   % target rank for truncation and min
eps_val    = 1e-4;                 
zeta_vals  = [1,2,3,4];           % sparsity parameters
k_vals     = 50:20:300;   % sweep sketch dim
num_runs   = 100;

num_z    = numel(zeta_vals);
num_k    = numel(k_vals);

%% -- Fixed Q and A --
Q      = [eye(trunc_rank); zeros(n - trunc_rank, trunc_rank)];  % n×k
A_full = Q * Q' + eps_val * eye(n);                              % n×n

%% -- Preallocate storage --
sigmin_gauss = zeros(num_runs, num_k);
sigmin_zeta  = zeros(num_runs, num_k, num_z);
sigmin_iid   = zeros(num_runs, num_k, num_z);

err_gauss = zeros(1, num_k, num_runs);
err_zeta  = zeros(1, num_k, num_runs, num_z);
err_iid   = zeros(1, num_k, num_runs, num_z);

%% -- σ₋min experiments --
fprintf('Sigma-min on fixed Q (identity15)\n');
for zi = 1:num_z
    zeta = zeta_vals(zi);
    for ik = 1:num_k
        k = k_vals(ik);
        fprintf(' Task (ζ=%d, k=%d)\n', zeta, k);
        for run = 1:num_runs
            % Gaussian sketch Ω ∈ ℝ^{n×k}
            OmeG                         = randn(n, k) / sqrt(k);
            sigmin_gauss(run, ik)       = computeSigmin(Q, OmeG);

            % sparse-sign sketch Ω ∈ ℝ^{n×k}
            OmeS                         = sparse_sign_isubcols(k,n, zeta)';
            sigmin_zeta(run, ik, zi)    = computeSigmin(Q, OmeS);

            % iid-column sketch Ω ∈ ℝ^{n×k}
            OmeI                         = sparse_iid(n, k, zeta);
            sigmin_iid(run, ik, zi)     = computeSigmin(Q, OmeI);
        end
    end
end

%% -- Low-rank approximation experiments --
fprintf('Low-rank on A ∈ ℝ^{n×n}, varying sketch dim k\n');
normA = norm(A_full, 'fro');
for zi = 1:num_z
    zeta = zeta_vals(zi);
    for ik = 1:num_k
        k = k_vals(ik);
        fprintf(' Task (ζ=%d, k=%d)\n', zeta, k);
        for run = 1:num_runs
            % Gaussian sketch Ω ∈ ℝ^{n×k}
            OmeG                         = randn(n, k);
            Qg                           = orth(A_full * OmeG);
            err_gauss(1, ik, run)        = compute_error_mat(A_full, Qg, normA);

            % sparse-sign sketch Ω ∈ ℝ^{n×k}
            OmeS                         = sparse_sign_isubcols(k,n, zeta)';
            Qs                           = orth(A_full * OmeS);
            err_zeta(1, ik, run, zi)     = compute_error_mat(A_full, Qs, normA);

            % iid-column sketch Ω ∈ ℝ^{n×k}
            OmeI                         = sparse_iid(n, k, zeta);
            Qi                           = orth(A_full * OmeI);
            err_iid(1, ik, run, zi)      = compute_error_mat(A_full, Qi, normA);
        end
    end
end

%% -- Save results --
save('combined_zeta_with_iid_kvary.mat', ...
     'k_vals','sigmin_gauss','sigmin_zeta','sigmin_iid','zeta_vals', ...
     'err_gauss','err_zeta','err_iid');
fprintf('Data saved (%.2f s elapsed)\n', toc(t_start));

%% ------------------------------------------------------------------------
function smin = computeSigmin(Q, Ome)
    s = svd(Q' * Ome, 'econ');
    smin = s(end);
end

function err = compute_error_mat(A, Q, normA)
    [U,S,V] = svd(Q' * A, 'econ');
    A_k     = Q * U * S * V';
    err     = norm(A - A_k, 'fro') / normA;
end

function S = sparse_iid(d, k, zeta)
    p      = zeta / k;
    [rows,cols]  = find(rand(d, k) < p);
    V      = 2*randi([0,1], numel(rows), 1) - 1;
    S      = sqrt(k/zeta)*sparse(rows, cols, V, d, k);
end
