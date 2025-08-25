clear; clc;

nCores = feature('numcores');
maxNumCompThreads(nCores);

% — parameters —
pow_range = 2:14;                    
rs        = 2.^pow_range;        % r = 4,8,...,1024
ell_list  = 1:3:10;          
num_runs  = 2;
dist      = 'gaussian';
sphere    = false;

% — storage for smin/smax of Q'*Om —
n_r   = numel(rs);
n_ell = numel(ell_list);
alphas = zeros(n_r, n_ell, num_runs);  
betas  = zeros(n_r, n_ell, num_runs);   

rng(0);
fprintf('Starting parameter sweep: %d ell values × %d r values × %d runs each\n', ...
        n_ell, n_r, num_runs);

for ir = 1 : n_r
    r = rs(ir);
    d = 2^14;                  
    k = 2*r;                    

    if exist('fwht_mex','file') == 3
        H = fwht_mex(eye(d));
    else
        H = hadamard(d);
    end
    Q = H(:, 1:r);
    Q = Q ./ vecnorm(Q,2,1);

    for run = 1 : num_runs
        for iell = 1 : n_ell
            ell = ell_list(iell);

            Om = krSketch(d, k, ell, dist, sphere);   
            S  = svd(Q' * Om, 'econ');               
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