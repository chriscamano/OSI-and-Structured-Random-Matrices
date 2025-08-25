%% combined_zeta_tests.m
clear; clc; t_start = tic;

% -- Parameters --
r         = 10;                % ambient exponent
n         = 2^r;               % ambient dimension
trunc_rank = 50;               % subspace / low-rank target
eps_val   = 1e-4;
zeta_vals = [1,2,3,4];        % sparsity parameters
d_sigma   = unique(round(logspace(log10(trunc_rank), log10(n), 16)));
d_error   = trunc_rank : 10 : 200;
num_runs  = 5;

num_zeta = numel(zeta_vals);
num_dsig = numel(d_sigma);
num_der  = numel(d_error);

% -- Build two random subspaces via QR of Gaussians --
Q_list  = cell(1,2);
Q_names = {'gaussQR1','gaussQR2'};
for qi = 1:2
    [Q_list{qi},~] = qr(randn(n, trunc_rank), 0);
end

% -- Build two PSD matrices: A_i = Q_i * Q_i' + eps_val * I --
A_list  = cellfun(@(Q) Q*Q' + eps_val*eye(n), Q_list, 'UniformOutput', false);
A_names = {'A_gaussQR1','A_gaussQR2'};

% -- Preallocate storage --
sigmin_gauss = zeros(num_runs, num_dsig, numel(Q_list));
sigmin_zeta  = zeros(num_runs, num_dsig, numel(Q_list), num_zeta);

err_gauss = zeros(numel(A_list), num_der, num_runs);
err_zeta  = zeros(numel(A_list), num_der, num_runs, num_zeta);

% -- Sigma-min experiments --
totalTasks1 = numel(Q_list) * num_zeta * num_dsig;
task1 = 0;
for qi = 1:numel(Q_list)
    Q = Q_list{qi};
    fprintf('Sigma-min on Q = %s\n', Q_names{qi});
    for zi = 1:num_zeta
        zeta = zeta_vals(zi);
        for id = 1:num_dsig
            d = d_sigma(id);
            task1 = task1 + 1;
            fprintf(' Task %d/%d: zeta=%d, d=%d\n', task1, totalTasks1, zeta, d);
            for run = 1:num_runs
                % Gaussian sketch
                OmeG = randn(n, d) / sqrt(d);
                sigmin_gauss(run, id, qi) = computeSigmin(Q, OmeG);
                % sparse-sign i-subcols sketch
                OmeS = sparse_sign_isubcols(d, n, zeta)';
                sigmin_zeta(run, id, qi, zi) = computeSigmin(Q, OmeS);
            end
        end
    end
end

% -- Low-rank approximation experiments --
totalTasks2 = numel(A_list) * num_zeta * num_der;
task2 = 0;
for ai = 1:numel(A_list)
    A_full = A_list{ai};
    normA  = norm(A_full, 'fro');
    fprintf('Low-rank on A = %s\n', A_names{ai});
    for zi = 1:num_zeta
        zeta = zeta_vals(zi);
        for id = 1:num_der
            d = d_error(id);
            task2 = task2 + 1;
            fprintf(' Task %d/%d: zeta=%d, d=%d\n', task2, totalTasks2, zeta, d);
            for run = 1:num_runs
                % Gaussian
                OmeG = randn(n, d);
                [Qg, ~] = qr(A_full * OmeG, 0);
                err_gauss(ai, id, run) = compute_error_mat(A_full, Qg, normA);
                % sparse-sign i-subcols
                OmeS = sparse_sign_isubcols(d, n, zeta)';
                [Qs, ~] = qr(A_full * OmeS, 0);
                err_zeta(ai, id, run, zi) = compute_error_mat(A_full, Qs, normA);
            end
        end
    end
end

% -- Save results --
save('combined_zeta.mat', ...
     'd_sigma','sigmin_gauss','sigmin_zeta','Q_names','zeta_vals', ...
     'd_error','err_gauss','err_zeta','A_names');
fprintf('Data saved (%.2f s elapsed)\n', toc(t_start));

% -- Helper functions --
function smin = computeSigmin(Q, Ome)
    M = Q' * Ome;
    s = svd(M, 'econ');
    smin = s(end);
end

function err = compute_error_mat(A, Q, normA)
    B = Q' * A;
    [U, S, V] = svd(B, 'econ');
    A_k = Q * U * S * V';
    err = norm(A - A_k, 'fro') / normA;
end
