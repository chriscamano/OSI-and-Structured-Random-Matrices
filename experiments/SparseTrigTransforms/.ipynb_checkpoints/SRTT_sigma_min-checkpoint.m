% combined_experiment_srtt.m
clear; clc; t_start = tic;

%% — Parameters —
r           = 10;               % ambient exponent
n           = 2^r;              % ambient dimension
trunc_rank  = 50;               % target subspace dim
eps_val     = 1e-4;
rounds      = 1;                % # of sign+transform rounds
num_runs    = 5;                % adjust as needed

%% — Transforms to test —
transforms   = {'dct','fwht','fourier'};
nTrans       = numel(transforms);

%% — Build two random subspaces via QR of Gaussians —
Q_list  = cell(1,2);
Q_names = {'gaussQR1','gaussQR2'};
for qi = 1:2
    [Q,~]      = qr(randn(n, trunc_rank), 0);
    Q_list{qi} = Q;
end

%% — Build PSD matrices A_i = Q_i Q_i' + eps·I —
A_list  = cellfun(@(Q) Q*Q' + eps_val*eye(n), Q_list, 'UniformOutput',false);
A_names = {'A_gaussQR1','A_gaussQR2'};

%% — σ_min experiment setup —
d_sigma       = unique(round(logspace(log10(trunc_rank), log10(n), 16)));
num_d         = numel(d_sigma);
sigmin_gauss  = zeros(num_runs, num_d, numel(Q_list));
sigmin_srtt   = zeros(num_runs, num_d, numel(Q_list), nTrans);

%% — Run σ_min experiments —
for qi = 1:numel(Q_list)
    Q = Q_list{qi};
    fprintf('σₘᵢₙ on Q = %s\n', Q_names{qi});
    for run = 1:num_runs
        for id = 1:num_d
            d = d_sigma(id);

            % Gaussian sketch
            Ome = randn(n, d) / sqrt(d);
            sigmin_gauss(run, id, qi) = computeSigmin(Q, Ome);

            % SRTT sketches
            for ti = 1:nTrans
                T = transforms{ti};
                St = srtt_explicit(n, d, rounds, T);
                sigmin_srtt(run, id, qi, ti) = computeSigmin(Q, St);
            end
        end
    end
end

%% — Low‐rank approximation experiment setup —
d_error    = trunc_rank : 10 : 200;  
num_d_e    = numel(d_error);
err_gauss  = zeros(numel(A_list), num_d_e, num_runs);
err_srtt   = zeros(numel(A_list), num_d_e, num_runs, nTrans);

labels = ['Gaussian', transforms];

%% — Run low‐rank approximation experiments —
for ai = 1:numel(A_list)
    A_full = A_list{ai};
    normA  = norm(A_full,'fro');
    fprintf('Low-rank on A = %s\n', A_names{ai});
    for run = 1:num_runs
        for id = 1:num_d_e
            d = d_error(id);

            % Gaussian
            Ome = randn(n, d);
            [Qg, ~] = qr(A_full * Ome, 0);
            err_gauss(ai, id, run) = compute_error_mat(A_full, Qg, normA);

            % SRTT sketches
            for ti = 1:nTrans
                T = transforms{ti};
                Ome_s = srtt_explicit(n, d, rounds, T);
                [Qs, ~] = qr(A_full * Ome_s, 0);
                err_srtt(ai, id, run, ti) = compute_error_mat(A_full, Qs, normA);
            end
        end
    end
end

%% — Save results —
save('combined_srtt.mat', ...
     'd_sigma','sigmin_gauss','sigmin_srtt','Q_names','transforms', ...
     'd_error','err_gauss','err_srtt','A_names','labels');
fprintf('Data saved (%.2f s elapsed)\n', toc(t_start));


%% — Helper functions — 
function s_min = computeSigmin(Q, Ome)
    M     = Q' * Ome;
    s     = svd(M,'econ');
    s_min = s(end);
end

function err = compute_error_mat(A, Q, normA)
    B     = Q' * A;
    [U,S,V]= svd(B,'econ');
    A_k   = Q * U * S * V';
    err   = norm(A - A_k,'fro') / normA;
end
