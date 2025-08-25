% clear; clc;
% 
% % Use all cores
% nCores = feature('numcores');
% maxNumCompThreads(nCores);
% 
% % — parameters —
% pow_range = 10:12;
% rs        = 2.^pow_range;        % r = 4,8,...,1024
% ell_list  = 10:1:12;              % KR depths
% num_runs  = 100;
% dist      = 'gaussian';
% sphere    = false;
% k_steps   = 200;                  % Lanczos steps (clipped to <= r)
% 
% % — storage for σ_min/σ_max of Q'*Ω —
% n_r   = numel(rs);
% n_ell = numel(ell_list);
% alphas = zeros(n_r, n_ell, num_runs);   % σ_min
% betas  = zeros(n_r, n_ell, num_runs);   % σ_max
% 
% rng(0);
% fprintf('Starting parameter sweep: %d ell values × %d r values × %d runs each\n', ...
%         n_ell, n_r, num_runs);
% 
% for ir = 1:n_r
%     r = rs(ir);
%     d = 2^12;                   % ambient dimension (power of two for Hadamard)
%     k = 2*r;                    % embedding dimension
% 
%     % Orthonormal Q (Hadamard columns)
%     if exist('fwht_mex','file') == 3
%         H = fwht_mex(eye(d));
%     else
%         H = hadamard(d);
%     end
%     Q = H(:, 1:r);
%     Q = Q ./ vecnorm(Q,2,1);
% 
%     for run = 1:num_runs
%         for iell = 1:n_ell
%             ell = ell_list(iell);
% 
%             % Ω ∈ R^{d×k}
%             Om = krSketch(d, k, ell, dist, sphere);
% 
%             % Define Mx = (Q'ΩΩ'Q) x as a matvec; this avoids forming A explicitly.
%             M = @(x) Q' * (Om * (Om' * (Q * x)));
% 
%             % Random start vector on the unit sphere in R^r
%             z0 = randn(r,1); z0 = z0 / norm(z0);
% 
%             % Clip k_steps to r
%             m  = min(k_steps, r);
% 
%             % Lanczos tridiagonalization
%             [Qlan, T] = lanczos(M, z0, m); %#ok<ASGLU>
% 
%             % Ritz values approximate eigenvalues of A
%             theta = eig(T);
% 
%             % Numerical guard against tiny negative values from roundoff
%             lam_min = max(min(theta), 0);
%             lam_max = max(theta);
% 
%             % Convert back to singular values
%             alphas(ir, iell, run) = sqrt(lam_min);
%             betas (ir, iell, run) = sqrt(lam_max);
% 
%             clear Om;
%         end
% 
%         if mod(run,10)==0
%             fprintf('  r = %4d: completed %3d/%3d runs\n', r, run, num_runs);
%         end
%     end
% 
%     clear H Q
% end
% 
% % — save results —
% save('krsketch_sigma_extrema_gauss_lanczos.mat', ...
%      'rs', 'ell_list', 'num_runs', 'dist', 'sphere', 'k_steps', ...
%      'alphas', 'betas');
% 
% fprintf('All data saved to krsketch_sigma_extrema_gauss_lanczos.mat\n');

clear; clc;

% Use all cores
nCores = feature('numcores');
maxNumCompThreads(nCores);

% — parameters —
pow_range = 2:14;                    
rs        = 2.^pow_range;        % r = 4,8,...,1024
ell_list  = 1:3:10;              % KR depths
num_runs  = 2;
dist      = 'gaussian';
sphere    = false;

% — storage for σ_min/σ_max of Q'*Om —
n_r   = numel(rs);
n_ell = numel(ell_list);
alphas = zeros(n_r, n_ell, num_runs);   % KR: σ_min
betas  = zeros(n_r, n_ell, num_runs);   % KR: σ_max

rng(0);
fprintf('Starting parameter sweep: %d ell values × %d r values × %d runs each\n', ...
        n_ell, n_r, num_runs);

for ir = 1 : n_r
    r = rs(ir);
    d = 2^14;                   % ambient dimension (power of two for Hadamard)
    k = 2*r;                    % embedding dimension

    if exist('fwht_mex','file') == 3
        H = fwht_mex(eye(d));
    else
        H = hadamard(d);
    end
    Q = H(:, 1:r);
    Q = Q ./ vecnorm(Q,2,1);

    for run = 1 : num_runs
        % KR sketches across ell
        for iell = 1 : n_ell
            ell = ell_list(iell);

            Om = krSketch(d, k, ell, dist, sphere);   % Ω ∈ R^{d×k}
            S  = svd(Q' * Om, 'econ');                % singular values of r×k
            alphas(ir, iell, run) = S(end);
            betas (ir, iell, run) = S(1);

            clear Om;
        end
        if mod(run,10)==0
            fprintf('  r = %4d: completed %3d/%3d runs\n', r, run, num_runs);
        end
    end
    clear H Q
end

% — save results (no Gaussian baseline) —
save('krsketch_sigma_extrema_gauss2.mat', ...
     'rs', 'ell_list', 'num_runs', 'dist', 'sphere', ...
     'alphas', 'betas');

fprintf('All data saved to krsketch_sigma_extrema.mat\n');

function [Q,T] = lanczos(M,z,k)
% Input:  Function M() computing matrix–vector products, vector z,
%         and number of steps k
% Output: Matrix Q with orthonormal columns and tridiagonal matrix T

Q = zeros(length(z),k+1);
T = zeros(k+1);
Q(:,1) = z / norm(z);

for i = 1:k
    if i == 1
        Q(:,2) = M(Q(:,1));
    else
        Q(:,i+1) = M(Q(:,i)) - T(i,i-1)*Q(:,i-1);
    end
    T(i,i)       = Q(:,i+1)'*Q(:,i);
    Q(:,i+1)     = Q(:,i+1) - T(i,i)*Q(:,i);
    T(i,i+1)     = norm(Q(:,i+1));
    T(i+1,i)     = T(i,i+1);
    Q(:,i+1)     = Q(:,i+1) / T(i,i+1);
end

T = T(1:k,1:k);
Q = Q(:,1:k);
end 