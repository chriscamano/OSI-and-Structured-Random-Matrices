nCores = feature('numcores');
maxNumCompThreads(nCores);
%— parameters —
rs         = round(logspace(1, 7, 20));  
zeta_list  = [1,2,3,4,8];            % sparsity parameters to sweep
num_runs   = 30;                     % number of independent Lanczos runs
k_steps    = 200;                    % Lanczos steps

%— preallocate storage —
n_r        = numel(rs);
n_z        = numel(zeta_list);
alphas     = zeros(n_r, n_z, num_runs);
betas      = zeros(n_r, n_z, num_runs);

fprintf('Starting parameter sweep: %d zeta values × %d r values × %d runs each\n', ...
        n_z, n_r, num_runs);

%— main sweep —
for iz = 1 : n_z
    zeta = zeta_list(iz);
    fprintf(' zeta = %d\n', zeta);
    for ir = 1 : n_r
        r = rs(ir);
        fprintf('  r = %g\n', r);
        for run = 1 : num_runs
            S = sparse_sign_isubcols(2*r, r, zeta);
            z0 = randn(r,1);
            [~, T] = lanczos(@(v) S'*(S*v), z0, k_steps);

            ev = eig(T);
            alphas(ir,iz,run) = min(ev);
            betas (ir,iz,run) = max(ev);
            
            % print raw values for this run
            fprintf('    run %2d → min Ritz = %.5e, max Ritz = %.5e\n', ...
                    run, alphas(ir,iz,run), betas(ir,iz,run));
        end
        fprintf('  completed %d runs for r = %g, zeta = %d\n', num_runs, r, zeta);
    end
    fprintf(' completed all runs for zeta = %d\n', zeta);
end

%— save results —
save('sigma_min_max_results.mat', ...
     'rs', 'zeta_list', 'num_runs', 'k_steps', ...
     'alphas', 'betas');
fprintf('All data saved to sigma_min_max_results.mat\n');


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