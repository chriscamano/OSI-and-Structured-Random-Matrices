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

%% -------- GPE parameters --------
Nx               = 256;       % grid size
L                = 10;        % half-domain on each axis
dt               = 0.001;     % time step
nstep            = 35000;     % total steps
plotEvery        = 1;         % record every step

g                = 150;       % interaction 
omega            = 0.8;
V0               = 100;       % barrier height
rs               = 3.0;       % stir radius
stirOmega        = 0.7;       % stir frequency
sigStir          = 1;         % stir width
dx               = 2 * L / Nx;
Ngrid            = Nx^2;
nSnap            = floor(nstep/plotEvery);

%% -------- Sketching parameters --------
zeta             = 4;         % sparsity for sparse-sign
oversample       = 5;         % oversampling for randomized range finder
max_sketch_rank  = 10000;

%% -------- generate A0 --------
fprintf('Generating A0 directly...\n');
[A0,~] = gross_pitaevskii(Nx, L, dt, nstep, plotEvery, ...
                          g, omega, V0, rs, stirOmega, sigStir);

% Choose precision: 'single' or 'double'
precisionType = 'double';  % change to 'double' for double-precision

% cast to chosen precision
A0    = cast(A0, precisionType);
normA = cast(norm(A0, 'fro'), precisionType);
% At= A0';
%% -------- compute full (dense) SVD baseline --------
% fprintf('Computing full (dense) SVD of A0 for baseline...\n');
% t0 = tic;
% [Ufull, Sfull, Vfull] = svd(A0, 'econ');
% t_dense = toc(t0);
% fprintf('  → dense SVD done in %.3f s\n', t_dense);
%% -------- Prepare for Experiment --------
modeList        = round(logspace(log10(100), log10(max_sketch_rank), 10));
nModes          = numel(modeList);

times_gnsvd_ss    = zeros(1, nModes, precisionType);
errs_gnsvd_ss     = zeros(1, nModes, precisionType);
times_gnsvd_gauss = zeros(1, nModes, precisionType);
errs_gnsvd_gauss  = zeros(1, nModes, precisionType);

times_gn_ss     = zeros(1, nModes, precisionType);
errs_gn_ss      = zeros(1, nModes, precisionType);
times_gn_gauss  = zeros(1, nModes, precisionType);
errs_gn_gauss   = zeros(1, nModes, precisionType);

times_rsvd_ss    = zeros(1, nModes, precisionType);
errs_rsvd_ss     = zeros(1, nModes, precisionType);
times_rsvd_gauss = zeros(1, nModes, precisionType);
errs_rsvd_gauss  = zeros(1, nModes, precisionType);

times_qb_ss      = zeros(1, nModes, precisionType);
errs_qb_ss       = zeros(1, nModes, precisionType);
times_qb_gauss   = zeros(1, nModes, precisionType);
errs_qb_gauss    = zeros(1, nModes, precisionType);

%% -------- Main Experiment --------
for idx = 1:nModes

    r = modeList(idx);
    k = r + oversample;
    p = ceil(1.5 * k);

    fprintf('Iteration %d/%d: target rank k = %d\n', idx, nModes, k);

    %–––––––– Generate sparse-sign sketches ––––––––
    t_gen_Om_ss = tic;
    Om_ss = cast(sparse_sign_isubcols(k, size(A0,2), zeta)', precisionType);
    t_gen_Om_ss = toc(t_gen_Om_ss);

    t_gen_Psi_ss = tic;
    Psi_ss = cast(sparse_sign_isubcols(p, size(A0,1), zeta)', precisionType);
    t_gen_Psi_ss = toc(t_gen_Psi_ss);
    % 
    %–––––––– Compute Sketches AX_ss, YA_ss ––––––––
    % t_AO_ss   = tic; AX_ss   = A0*Om_ss;       t_AO_ss   = toc(t_AO_ss);
    % fprintf('Basic matlab * ✓ A*Ω_ss complete (%.3fs)\n', t_AO_ss);
    %  t_AO_ss   = tic; AX_ss   = sparseMM(A0,Om_ss);       t_AO_ss   = toc(t_AO_ss);
    % fprintf('custom c++ * ✓ A*Ω_ss complete (%.3fs)\n', t_AO_ss);
    %  t_AO_ss   = tic; AX_ss   = sparseMM2(A0,Om_ss);       t_AO_ss   = toc(t_AO_ss);
    % fprintf('custom c++ * ✓ A*Ω_ss complete (%.3fs)\n', t_AO_ss);
     t_AO_ss   = tic; AX_ss   = (Om_ss' * A0')';       t_AO_ss   = toc(t_AO_ss);
    fprintf('Transpose Trick ✓ A*Ω_ss complete (%.3fs)\n', t_AO_ss);
    % t_AO_ss = tic;
    % AX_ss = sfmult( A0, Om_ss, ...
    %                 0, 0, ...   % at=0, ac=0 → use A (no transpose/conj)
    %                 0, 0, ...   % xt=0, xc=0 → use x  (no transpose/conj)
    %                 0, 0 );     % yt=0, yc=0 → return y  (no transpose/conj)
    % t_AO_ss = toc(t_AO_ss);
    % fprintf('sfmult ✓ A*Ω_ss complete (%.3fs)\n\n', t_AO_ss);


    t_PsiA_ss = tic;  YA_ss=   Psi_ss' * A0;          t_PsiA_ss = toc(t_PsiA_ss);
    fprintf('Basic matlab *  ✓ Ψ_ss''*A complete (%.3fs)\n', t_PsiA_ss);
    % t_PsiA_ss = tic; 
    % YA_ss = sfmult( Psi_ss, A0, ...
    %             1, 0, ...   % at=1 (use A.'),   ac=0 (no conj)
    %             0, 0, ...   % xt=0, xc=0        (use x as-is)
    %             0, 0 );     % yt=0, yc=0        (return y as-is)
    % t_PsiA_ss = toc(t_PsiA_ss);
    % fprintf('sfmult *  ✓ Ψ_ss''*A complete (%.3fs)\n', t_PsiA_ss);
    % 

    %––– GN-SVD (sparse-sign) –––
    [t_ss_core, errs_gnsvd_ss(idx)] = generalized_Nystrom_SVD_core(...
        AX_ss, YA_ss, Psi_ss, k, A0, normA);
    times_gnsvd_ss(idx) = t_gen_Om_ss + t_gen_Psi_ss + t_AO_ss + t_PsiA_ss + t_ss_core;
    fprintf('  ✓ GN–SVD (SS) complete (%.3fs)\n', t_ss_core);

    %––– GN-direct (sparse-sign) –––
    [t_gn_ss, errs_gn_ss(idx)] = generalized_Nystrom_core(...
        AX_ss, YA_ss, Psi_ss, A0, normA);
    times_gn_ss(idx) = t_gen_Om_ss + t_gen_Psi_ss + t_AO_ss + t_PsiA_ss + t_gn_ss;
    fprintf('  ✓ GN–direct (SS) complete (%.3fs)\n', t_gn_ss);

    %––– RSVD & QB (sparse-sign) –––
    [t_qb_ss, t_rsvd_ss, errs_qb_ss(idx), errs_rsvd_ss(idx)] = rsvd_core(...
        AX_ss, A0, normA);
    times_qb_ss(idx)    = t_gen_Om_ss + t_AO_ss + t_qb_ss;
    times_rsvd_ss(idx)  = t_gen_Om_ss + t_AO_ss + t_rsvd_ss;
    fprintf('  ✓ RSVD & QB (SS) complete (%.3f / %.3fs)\n', t_qb_ss, t_rsvd_ss);

    clear AX_ss YA_ss Om_ss Psi_ss At;

    %% –––––––– Generate Gaussian sketches ––––––––
    t_gen_Om_g = tic;
    Om_g  = cast(randn(size(A0,2), k), precisionType);
    t_gen_Om_g = toc(t_gen_Om_g);

    t_gen_Psi_g = tic;
    Psi_g = cast(randn(size(A0,1), p), precisionType);
    t_gen_Psi_g = toc(t_gen_Psi_g);

    %–––––––– Compute Sketches AX_g, YA_g ––––––––
    t_AO_g   = tic; AX_g   = A0 * Om_g;      t_AO_g   = toc(t_AO_g);
    fprintf('  ✓ A*Ω_g complete (%.3fs)\n', t_AO_g);
    t_PsiA_g = tic; YA_g   = Psi_g' * A0;    t_PsiA_g = toc(t_PsiA_g);
    fprintf('  ✓ Ψ_g''*A complete (%.3fs)\n', t_PsiA_g);

    %––– GN-SVD (Gaussian) –––
    [t_g_core, errs_gnsvd_gauss(idx)] = generalized_Nystrom_SVD_core(...
        AX_g, YA_g, Psi_g, k, A0, normA);
    times_gnsvd_gauss(idx) = t_gen_Om_g + t_gen_Psi_g + t_AO_g + t_PsiA_g + t_g_core;
    fprintf('  ✓ GN–SVD (G) complete (%.3fs)\n', t_g_core);

    %––– GN-direct (Gaussian) –––
    [t_gn_g, errs_gn_gauss(idx)] = generalized_Nystrom_core(...
        AX_g, YA_g, Psi_g, A0, normA);
    times_gn_gauss(idx) = t_gen_Om_g + t_gen_Psi_g + t_AO_g + t_PsiA_g + t_gn_g;
    fprintf('  ✓ GN–direct (G) complete (%.3fs)\n', t_gn_g);

    %––– RSVD & QB (Gaussian) –––
    [t_qb_g, t_rsvd_g, errs_qb_gauss(idx), errs_rsvd_gauss(idx)] = rsvd_core(...
        AX_g, A0, normA);
    times_qb_gauss(idx)   = t_gen_Om_g + t_AO_g + t_qb_g;
    times_rsvd_gauss(idx) = t_gen_Om_g + t_AO_g + t_rsvd_g;
    fprintf('  ✓ RSVD & QB (G) complete (%.3f / %.3fs)\n', t_qb_g, t_rsvd_g);

    clear AX_g YA_g Om_g Psi_g;

    %––– compute and print speed-ups –––
    sp_svd  = times_gnsvd_gauss(idx)  / times_gnsvd_ss(idx);
    sp_gn   = times_gn_gauss(idx)     / times_gn_ss(idx);
    sp_rsvd = times_rsvd_gauss(idx)   / times_rsvd_ss(idx);
    sp_qb   = times_qb_gauss(idx)     / times_qb_ss(idx);

    fprintf([ ...
      '==========================\n' ...
      'k=%3d: SS[GNSVD %.3f/%.2e, GN %.3f/%.2e, ' ...
           'RSVD %.3f/%.2e, QB %.3f/%.2e]  ' ...
           'G[   SVD %.3f/%.2e, GN %.3f/%.2e, ' ...
           'RSVD %.3f/%.2e, QB %.3f/%.2e]\n  ' ...
      'Speedup(SS→G)[SVD %.2f×, GN %.2f×, ' ...
           'RSVD %.2f×, QB %.2f×]\n' ...
      '==========================\n\n' ...
    ], ...
      k, ...
      times_gnsvd_ss(idx), errs_gnsvd_ss(idx), ...
      times_gn_ss(idx), errs_gn_ss(idx), ...
      times_rsvd_ss(idx), errs_rsvd_ss(idx), ...
      times_qb_ss(idx), errs_qb_ss(idx), ...
      times_gnsvd_gauss(idx), errs_gnsvd_gauss(idx), ...
      times_gn_gauss(idx), errs_gn_gauss(idx), ...
      times_rsvd_gauss(idx), errs_rsvd_gauss(idx), ...
      times_qb_gauss(idx), errs_qb_gauss(idx), ...
      sp_svd, sp_gn, sp_rsvd, sp_qb ...
    );

end

%% -------- save sweep results --------
dataDir    = fullfile('data','POD');
resultFile = fullfile(dataDir,'POD_sweep_results.mat');
save(resultFile, 'modeList', 'times_*', 'errs_*');
fprintf('Saved POD sweep results to %s\n', resultFile);

%% -------- plot all methods --------
figure;
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

ax1 = nexttile;
semilogy(modeList, errs_gnsvd_ss,    '-o', ...
         modeList, errs_gn_ss,     '--o', ...
         modeList, errs_gnsvd_gauss, '-s', ...
         modeList, errs_gn_gauss,  '--s', ...
         modeList, errs_rsvd_ss,   '-x', ...
         modeList, errs_rsvd_gauss,'--x', ...
         modeList, errs_qb_ss,     '-d', ...
         modeList, errs_qb_gauss,  '--d');
xlabel('Number of modes k');
ylabel('Relative reconstruction error');
legend('GNSVD–SS','GN–SS','RSVD–SS','QB–SS', ...
       'GNSVD–G','GN–G','RSVD–G','QB–G','Location','best');
grid(ax1,'on');

ax2 = nexttile;
semilogy(modeList, times_gnsvd_ss,    '-o', ...
         modeList, times_gn_ss,     '--o', ...
         modeList, times_gnsvd_gauss, '-s', ...
         modeList, times_gn_gauss,  '--s', ...
         modeList, times_rsvd_ss,   '-x', ...
         modeList, times_rsvd_gauss,'--x', ...
         modeList, times_qb_ss,     '-d', ...
         modeList, times_qb_gauss,  '--d');
xlabel('Number of modes k');
ylabel('Wall-clock time (s)');
legend('GNSVD–SS','GN–SS','RSVD–SS','QB–SS', ...
       'GNSVD–G','GN–G','RSVD–G','QB–G','Location','best');
grid(ax2,'on');

%% -------- local functions --------
%{ 
To save memory and time, all relative low-rank approximation errors are
computed within the scope of their respective function, and sketches are
passed as arguments. For standalone versions, see code/low_rank_approximation.
%}
function [elapsed, relErr] = generalized_Nystrom_core(AX, YA, Psi, A, normA)
    t0 = tic;
    [Q, R] = qr(Psi' * AX, 0);    % small QR
    U      = AX / R;              % (n×k)
    V      = Q' * YA;             % (k×m)
    elapsed = toc(t0);
    relErr  = norm(A - U*V, 'fro') / normA;
end

function [elapsed, relErr] = generalized_Nystrom_SVD_core(AX, YA, Psi, k, A, normA)
    t0 = tic;
    [Q,~]    = qr(AX, 0);         % (n×k)
    [P, T]   = qr(YA', 0);        % (m×k)
    C        = (Psi' * Q) \ T';   % k×k
    [Uc, Sc, Vc] = svd(C, 'econ');
    U = Q * Uc(:,1:k);
    S = Sc(1:k,1:k);
    V = P * Vc(:,1:k);
    elapsed = toc(t0);
    relErr  = norm(A - U*S*V', 'fro') / normA;
end

function [t_qb, t_rsvd, relErrQB, relErrRSVD] = rsvd_core(AX, A, normA)
    t0 = tic;
    [Q, ~] = qr(AX, 0);           % form Q
    B      = Q' * A;              % project
    t_qb   = toc(t0);
    relErrQB  = norm(A - Q*B, 'fro') / normA;

    t1 = tic;
    [UU, S, V] = svd(B, 'econ');   % small SVD
    U          = Q * UU;           % lift
    t_rsvd     = t_qb + toc(t1);
    relErrRSVD = norm(A - U*S*V', 'fro') / normA;
end



% %% cached dense x sparse bench
% k_vals = 7000:500:10000;
% 
% % Preallocate timing arrays
% times_matlab      = zeros(size(k_vals));
% times_custom      = zeros(size(k_vals));
% times_custom2     = zeros(size(k_vals));
% times_transpose   = zeros(size(k_vals));
% times_transpose_pre = zeros(size(k_vals));
% times_sfmult      = zeros(size(k_vals));
% 
% A0_transpose = A0'; % Precompute A'
% 
% 
% %% Main Loop
% d = nstep;
% 
% for idx = 1:length(k_vals)
%     idx
%     k = k_vals(idx);
%     p = ceil(1.5 * k); % Just to match your code
% 
%     fprintf('\n=== k = %d ===\n', k);
% 
%     % Generate Omega
%     Om_ss = cast(sparse_sign_isubcols(k, d, zeta)', precisionType);
% 
%     % Method 1: Basic Matlab A * Omega_ss
%     t = tic;
%     AX_ss = A0 * Om_ss;
%     times_matlab(idx) = toc(t);
%     fprintf('Basic matlab * ✓ A*Ω_ss complete (%.3fs)\n', times_matlab(idx));
% 
%     % Method 2: Custom C++ sparseMM
%     t = tic;
%     AX_ss = sparseMM(A0, Om_ss);
%     times_custom(idx) = toc(t);
%     fprintf('Custom C++ sparseMM ✓ A*Ω_ss complete (%.3fs)\n', times_custom(idx));
%     % 
%     % % Method 3: Custom C++ sparseMM2
%     % t = tic;
%     % AX_ss = loopDenseSparse(A0, Om_ss);
%     % times_custom2(idx) = toc(t);
%     % fprintf('Custom C++ sparseMM2 ✓ A*Ω_ss complete (%.3fs)\n', times_custom2(idx));
% 
%     % Method 4: Transpose Trick
%     t = tic;
%     AX_ss = (Om_ss' * A0')';
%     times_transpose(idx) = toc(t);
%     fprintf('Transpose Trick ✓ A*Ω_ss complete (%.3fs)\n', times_transpose(idx));
% 
%     % Method 5: Transpose Trick + precompute A'
%     t = tic;
%     AX_ss = (Om_ss' * A0_transpose)';
%     times_transpose_pre(idx) = toc(t);
%     fprintf('Transpose Trick + precompute ✓ A*Ω_ss complete (%.3fs)\n', times_transpose_pre(idx));
% 
%     % Method 6: sfmult
%     t = tic;
%     AX_ss = sfmult( A0, Om_ss, ...
%                     0, 0, ...   % at=0, ac=0 → use A (no transpose/conj)
%                     0, 0, ...   % xt=0, xc=0 → use x  (no transpose/conj)
%                     0, 0 );     % yt=0, yc=0 → return y  (no transpose/conj)
%     times_sfmult(idx) = toc(t);
%     fprintf('sfmult ✓ A*Ω_ss complete (%.3fs)\n', times_sfmult(idx));
% 
% end
% 
% %% Plot
% figure;
% semilogy(k_vals, times_matlab, '-o', 'LineWidth', 2);
% hold on;
% semilogy(k_vals, times_custom, '-s', 'LineWidth', 2);
% % semilogy(k_vals, times_custom2, '-<', 'LineWidth', 2);
% semilogy(k_vals, times_transpose, '-d', 'LineWidth', 2);
% semilogy(k_vals, times_transpose_pre, '-p', 'LineWidth', 2);
% semilogy(k_vals, times_sfmult, '-^', 'LineWidth', 2);
% hold off;
% 
% grid on;
% xlabel('k (sketch width)');
% ylabel('Runtime (seconds, log scale)');
% title('Runtime of A * Omega_ss vs. k');
% legend('Matlab *', ...
%        'Custom C++ sparseMM', ...
%        'Transpose Trick', ...
%        'Transpose Trick + precompute', ...
%        'sfmult', ...
%        'Location', 'northwest');
% set(gca, 'FontSize', 12);