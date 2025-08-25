%% -------- Introduction --------
%{ 
The following Matlab script is an optimized driver for comparing
different low rank approximation algorithms for the task of compressing
the Gross Pitaevskii equation—a nonlinear PDE appearing in condensed
matter physics which describes the flow of Bose–Einstein Condensates
evolving over time. A number of minor optimizations are made to make this 
experiment scalable to large domain resolutions, as detailed below.
%}

clearvars; close all; clc;
maxNumCompThreads(feature('numcores'));
n_runs = 5; % number of independent runs

%% -------- GPE parameters --------
Nx        = 256;
L         = 16;
g         = 9000;
Omega_max = 0.9;
dtau      = 1e-4;
maxIter   = 40000;
tol       = 1e-6;
saveEvery = 1;

%% sketch parameters
zeta = 4; 
oversample = 5; 
max_sketch_rank = 10000;

%% generate A0 once
fprintf('Generating A0 directly...\n');
Snap = GPEsim(Nx, L, g, Omega_max, dtau, maxIter, tol, saveEvery);
fprintf('Formatting data\n');
realSnap = real(Snap);
imagSnap = imag(Snap);
clear Snap;

meanReal = mean(realSnap, 2);  
meanImag = mean(imagSnap, 2); 
realSnap = realSnap - meanReal; 
imagSnap = imagSnap - meanImag;

% if you need the stacked A0:
A0 = [ realSnap ; imagSnap ]; 
clear SnapStacked;
normA = norm(A0, 'fro');

%% build mode list and preallocate
modeList = round(logspace(log10(100), log10(max_sketch_rank), 16));
nModes   = numel(modeList);

times_gnsvd_ss    = zeros(n_runs, nModes);
errs_gnsvd_ss     = zeros(n_runs, nModes);
times_gnss        = zeros(n_runs, nModes);
errs_gnss         = zeros(n_runs, nModes);

times_gnsvd_gauss = zeros(n_runs, nModes);
errs_gnsvd_gauss  = zeros(n_runs, nModes);
times_gngauss     = zeros(n_runs, nModes);
errs_gngauss      = zeros(n_runs, nModes);

sp_svd_arr        = zeros(n_runs, nModes);
sp_gn_arr         = zeros(n_runs, nModes);

for run = 1:n_runs
    fprintf('=== Run %d/%d ===\n', run, n_runs);

    for i = 1:nModes
        r = modeList(i);
        k = r + oversample;
        p = ceil(1.5 * k);
        fprintf('  Mode %d/%d: target rank k = %d\n', i, nModes, k);

        %––– sparse-sign sketches –––
        t0 = tic;
          Om_ss  = sparse_sign_isubcols(k, size(A0,2), zeta)';
          Psi_ss = sparse_sign_isubcols(p, size(A0,1), zeta)';
        gen_ss = toc(t0);

        tAO = tic; AX_ss = (Om_ss' * A0')'; tAO = toc(tAO);
        tPA = tic; YA_ss = Psi_ss' * A0;  tPA = toc(tPA);

        [tcore, errs_gnsvd_ss(run,i)] = generalized_Nystrom_SVD_core(...
          AX_ss, YA_ss, Psi_ss, A0, normA);
        times_gnsvd_ss(run,i) = gen_ss + tAO + tPA + tcore;

        [tgn, errs_gnss(run,i)] = generalized_Nystrom_core(...
          AX_ss, YA_ss, Psi_ss, A0, normA);
        times_gnss(run,i) = gen_ss + tAO + tPA + tgn;

        clear AX_ss YA_ss Om_ss Psi_ss

        %––– Gaussian sketches –––
        t0 = tic;
          Om_g  = randn(size(A0,2), k);
          Psi_g = randn(size(A0,1), p);
        gen_g = toc(t0);

        tAO = tic; AX_g = A0 * Om_g;       tAO = toc(tAO);
        tPA = tic; YA_g = Psi_g' * A0;     tPA = toc(tPA);

        [tcore_g, errs_gnsvd_gauss(run,i)] = generalized_Nystrom_SVD_core(...
          AX_g, YA_g, Psi_g, A0, normA);
        times_gnsvd_gauss(run,i) = gen_g + tAO + tPA + tcore_g;

        [tgn_g, errs_gngauss(run,i)] = generalized_Nystrom_core(...
          AX_g, YA_g, Psi_g, A0, normA);
        times_gngauss(run,i) = gen_g + tAO + tPA + tgn_g;

        clear AX_g YA_g Om_g Psi_g

        %––– speedups & recap –––
        sp_svd_arr(run,i) = times_gnsvd_gauss(run,i) / times_gnsvd_ss(run,i);
        sp_gn_arr(run,i)  = times_gngauss(run,i)   / times_gnss(run,i);
        fprintf('  → Speedups: SVD %.2f×, GN %.2f×\n\n', ...
                sp_svd_arr(run,i), sp_gn_arr(run,i));
    end
end

%% save for later analysis
save('POD_multiRun_results.mat', 'modeList', ...
     'times_gnsvd_ss','errs_gnsvd_ss', ...
     'times_gnss','errs_gnss', ...
     'times_gnsvd_gauss','errs_gnsvd_gauss', ...
     'times_gngauss','errs_gngauss', ...
     'sp_svd_arr','sp_gn_arr');

%% -------- plot all methods --------
figure;
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

ax1 = nexttile;
semilogy(modeList, errs_gnsvd_ss,    '-o', ...
         modeList, errs_gnss,       '--o', ...
         modeList, errs_gnsvd_gauss, '-s', ...
         modeList, errs_gngauss,    '--s');
xlabel('Number of modes k');
ylabel('Relative reconstruction error');
legend('GNSVD–SS','GN–SS','GNSVD–G','GN–G','Location','best');
grid(ax1,'on');

ax2 = nexttile;
loglog(modeList, times_gnsvd_ss,    '-o', ...
         modeList, times_gnss,       '--o', ...
         modeList, times_gnsvd_gauss, '-s', ...
         modeList, times_gngauss,    '--s');
xlabel('Number of modes k');
ylabel('Wall-clock time (s)');
legend('GNSVD–SS','GN–SS','GNSVD–G','GN–G','Location','best');
grid(ax2,'on');

%% -------- local functions --------
function [elapsed, relErr] = generalized_Nystrom_core(AX, YA, Psi, A, normA)
    t0 = tic;
    [Q, R] = qr(Psi' * AX, 0);
    U      = AX / R;
    V      = Q' * YA;
    elapsed = toc(t0);
    relErr  = norm(A - U*V, 'fro') / normA;
end

function [elapsed, relErr] = generalized_Nystrom_SVD_core(AX, YA, Psi, A, normA)
    t0 = tic;
    [Q,~]  = qr(AX, 0);
    [P, T] = qr(YA', 0);

    % Stabilized pseudoinverse
    [U1, s1, V1] = svd(Psi' * Q, 'econ', 'vector');
    sigma_max    = s1(1);
    tol          = 5 * eps(sigma_max);
    r            = sum(s1 > tol);
    C            = V1(:,1:r) * ((U1(:,1:r)'*T') ./ s1(1:r));

    % SVD of core
    [Uc, S, Vc] = svd(C, 'econ');
    U = Q * Uc;
    V = P * Vc;

    elapsed = toc(t0);
    relErr  = norm(A - U*S*V', 'fro') / normA;
end