clear; clc; t_start = tic;

%% — Parameters for Q's and A's —
r           = 10;          
n           = 2^r;
l           = r;            % number of KR factors
trunc_rank  = 50;           % dimension of subspaces
eps_val     = 1e-4;

%% — Build subspaces Q1 (Hadamard) and Q2 (Kron–Gaussian) —
% Q1 via truncated Walsh–Hadamard
H  = fwht(eye(n));               
Q1 = H(:,1:trunc_rank);           
Q1 = Q1 ./ vecnorm(Q1,2,1);

%[Q2,~] = qr(krSketch(n,trunc_rank,l),0);
Q_temp = 1;            % start with the scalar 1 so kron(1, Q2) = Q2
for i = 1:l
    % draw a random 2×2 Gaussian and orthonormalize
    [Q2, ~] = qr(randn(2), 0);
    % tensor‐product with the accumulated basis
    Q_temp = kron(Q_temp, Q2);
end
Q2 = Q_temp(:,1:trunc_rank);
Q_list  = {Q1, Q2};
Q_names = {'hadamard','kron'};

A_list  = cellfun(@(Q) Q*Q' + eps_val*eye(n), Q_list, 'UniformOutput',false);
A_names = {'A_hadamard','A_kron'};

%% — σ_min experiment setup —
num_runs   = 100;
k_raw      = logspace(log10(trunc_rank), log10(n), 16);
ks_sigma   = unique(round(k_raw));
num_k      = numel(ks_sigma);
num_Q      = numel(Q_list);

% Preallocate
sigmin_gauss         = zeros(num_runs, num_k, num_Q);
sigmin_krs_gauss     = zeros(num_runs, num_k, num_Q);
sigmin_krs_sphere    = zeros(num_runs, num_k, num_Q);
sigmin_krs_gauss_C   = zeros(num_runs, num_k, num_Q);
sigmin_krs_sphere_C  = zeros(num_runs, num_k, num_Q);
sigmin_krs_radem     = zeros(num_runs, num_k, num_Q);
sigmin_krs_cradem    = zeros(num_runs, num_k, num_Q);
sigmin_krs_steinhaus = zeros(num_runs, num_k, num_Q); 

%% — Run σ_min experiments —
for qi = 1:num_Q
    Q = Q_list{qi};
    fprintf('σₘᵢₙ runs on Q = %s\n', Q_names{qi});
    for run = 1:num_runs
        for ik = 1:num_k
            k = ks_sigma(ik);
            
            Ome = randn(n,k)/sqrt(k);
            sigmin_gauss(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'gaussian',false);
            sigmin_krs_gauss(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'gaussian',true);
            sigmin_krs_sphere(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'cgaussian',false);
            sigmin_krs_gauss_C(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'cgaussian',true);
            sigmin_krs_sphere_C(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'rademacher',false);
            sigmin_krs_radem(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'crademacher',false);
            sigmin_krs_cradem(run,ik,qi) = computeSigmin(Q,Ome);
            
            Ome = krSketch(n,k,l,'steinhaus',false);  % new
            sigmin_krs_steinhaus(run,ik,qi) = computeSigmin(Q,Ome);
        end
    end
end

%% — Low-rank approximation experiment setup —
ks_error = trunc_rank:10:200;
numA     = numel(A_list);
numM     = 8;            % was 7, now includes Steinhaus
numK_e   = numel(ks_error);
errors   = zeros(numA, numM, numK_e, num_runs);

labels = { 'Gaussian', 'KR Gaussian', 'KR Sphere', ...
           'KR Complex Gaussian', 'KR Complex Sphere', ...
           'KR Rademacher', 'KR Complex Rademacher', ...
           'KR Steinhaus' };  % new

%% — Run low-rank approximation experiments —
for ai = 1:numA
    A_full = A_list{ai};
    normA  = norm(A_full,'fro');
    for run = 1:num_runs
        for ik = 1:numK_e
            k = ks_error(ik);
            for mi = 1:numM
                switch mi
                  case 1, Ome = randn(n,k);
                  case 2, Ome = krSketch(n,k,l,'gaussian',false);
                  case 3, Ome = krSketch(n,k,l,'gaussian',true);
                  case 4, Ome = krSketch(n,k,l,'cgaussian',false);
                  case 5, Ome = krSketch(n,k,l,'cgaussian',true);
                  case 6, Ome = krSketch(n,k,l,'rademacher',false);
                  case 7, Ome = krSketch(n,k,l,'crademacher',false);
                  case 8, Ome = krSketch(n,k,l,'steinhaus',false);  % new
                end
                Qtmp = orth(A_full * Ome);
                errors(ai,mi,ik,run) = compute_error_mat(A_full, Qtmp, normA);
            end
        end
    end
end

%% — Save all results —
save('KRcombined2.mat', ...
     'ks_sigma', 'sigmin_gauss', 'sigmin_krs_gauss', 'sigmin_krs_sphere', ...
     'sigmin_krs_gauss_C', 'sigmin_krs_sphere_C', 'sigmin_krs_radem', ...
     'sigmin_krs_cradem', 'sigmin_krs_steinhaus', ...  % new
     'Q_names', ...
     'ks_error', 'errors', 'labels', 'A_names');

elapsed = toc(t_start);
fprintf('All data saved to KRcombined2.mat (%.2f s elapsed)\n', elapsed);

%% — Helper functions — 
function s_min = computeSigmin(Q, Ome)
    M = Q' * Ome;
    s = svd(M, 'econ');
    s_min = s(end);
end

function err = compute_error_mat(A, Q, normA)
    B       = Q' * A;
    [U,S,V] = svd(B, 'econ');
    A_k     = Q * U * S * V';
    err     = norm(A - A_k, 'fro') / normA;
end