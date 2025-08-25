clear; clc; t_start = tic;
%% -- Parameters --
r          = 13;                   % ambient exponent
n          = 2^r;                  % ambient dimension = 1024
trunc_rank = 50;                   % target rank for truncation and min
eps_val    = 1e-4;                 
zeta_vals  = [1,2,3,4];            % sparsity parameters

% simple sketch-dimension steps
k_vals     = 50:10:300;            % sketch dimensions to sweep: 25,50,75,100
num_runs   = 100;

num_z = numel(zeta_vals);
num_k = numel(k_vals);

%% -- Fixed Q and A --
Q      = [eye(trunc_rank); zeros(n - trunc_rank, trunc_rank)];  % n×k
A_full = Q * Q' + eps_val * eye(n);                             % n×n

%% -- Preallocate storage for σ₋min --
sigmin_gauss = zeros(num_runs, num_k);
sigmin_zeta  = zeros(num_runs, num_k, num_z);
sigmin_iid   = zeros(num_runs, num_k, num_z);

%% -- σ₋min experiments --
fprintf('Sigma-min on fixed Q\n');
for zi = 1:num_z
    gamma = zeta_vals(zi);
    for ik = 1:num_k
        k = k_vals(ik);
        fprintf(' Task (ζ=%d, k=%d)\n', gamma, k);
        for run = 1:num_runs
            % Gaussian sketch Ω ∈ ℝ^{n×k}
            OmeG                   = randn(n, k) / sqrt(k);
            sigmin_gauss(run, ik) = computeSigmin(Q, OmeG);

            % sparse-sign sketch Ω ∈ ℝ^{n×k}
            OmeS                      = sparse_sign_isubcols(k, n, gamma)';
            sigmin_zeta(run, ik, zi) = computeSigmin(Q, OmeS);

            % iid-column sketch Ω ∈ ℝ^{n×k}
            OmeI                     = sparse_iid(n, k, gamma);
            sigmin_iid(run, ik, zi)  = computeSigmin(Q, OmeI);
        end
    end
end

%% -- Save results --
% k_vals     = 50:20:300;   % this should be [50 70 90 … 290]

save('SSfailuremode2_sigmaMin.mat', ...
     'k_vals','sigmin_gauss','sigmin_zeta','sigmin_iid','zeta_vals');
fprintf('Data saved (%.2f s elapsed)\n', toc(t_start));

%% ------------------------------------------------------------------------
function smin = computeSigmin(Q, Ome)
    s = svd(Q' * Ome, 'econ');
    smin = s(end);
end

function S = sparse_iid(d, k, zeta)
    % Returns a d×k sketching matrix with zeta nnz per row in expectation
    p      = zeta / k;
    [rows, cols] = find(rand(d, k) < p);
    V      = 2*randi([0,1], numel(rows), 1) - 1;
    S      = sqrt(k/zeta) * sparse(rows, cols, V, d, k);
end
