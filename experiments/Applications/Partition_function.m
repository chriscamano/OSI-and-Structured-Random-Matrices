clear; clc;
rng(42);
%% Parameters
L           = 16;                     % number of spins
beta_levels = [.5, .75, 1];           % three betas of interest
n           = 2^L;                    % matrix dimension
num_runs    = 30;                     % number of repeats per beta
% m_vals      = round(logspace(log10(10), log10(50), num_pts));
m_vals = 10:10:80;
m_vals      = unique(m_vals + mod(m_vals,2));
M           = numel(m_vals);
ell         = L;
num_methods = 4;                     

%% Preallocate
B              = numel(beta_levels);
errs           = zeros(B, num_runs, M, num_methods);
times          = zeros(B, num_runs, M, num_methods);
true_tr        = zeros(B,1);
spectral_decay = cell(B,1);

%% Build Hamiltonian 
h       = 10;
H       = tfim(L, h);
b_shift = -(1+h)*L;
H_shift = H - b_shift * speye(n,n);

%% Main loops
for bi = 1:B
    beta = beta_levels(bi);
    % exact spectrum & trace
    d          = tfim_eigs(L, h);
    decay      = exp(-beta*(d - b_shift));
    spectral_decay{bi} = sort(decay,'descend');
    true_tr(bi)        = sum(decay);

    Afun = @(X) expmv(1, -beta * H_shift, X);
    fprintf('\n*** β = %.2f (true trace = %.4e) ***\n', beta, true_tr(bi));

    for run = 1:num_runs
        fprintf(' Run %d/%d\n', run, num_runs);
        for idx = 1:M
            k = m_vals(idx);
            fprintf('  k = %3d:', k);

            %% 1) Nyström++ (KR)
            t0 = tic;
            Om = krSketch(n, k/2, ell, "gaussian",true);
            Psi = krSketch(n, k/2, ell, "gaussian",true);
            Z1 = nystrompp(n, Afun, Om, Psi);
            times(bi,run,idx,1) = toc(t0);
            errs(bi,run,idx,1)  = abs(Z1 - true_tr(bi)) / true_tr(bi);

            %% 2) Hutchinson (KR)
            t0 = tic;
            Z2 = hutch_KR(n, Afun, k, ell);
            times(bi,run,idx,2) = toc(t0);
            errs(bi,run,idx,2)  = abs(Z2 - true_tr(bi)) / true_tr(bi);

            %% 3) X-Nyström (KR)
            t0 = tic;
            [Z3,~] = xnystrace(Afun, n, k, 'kr', ell);
            times(bi,run,idx,3) = toc(t0);
            errs(bi,run,idx,3)  = abs(Z3 - true_tr(bi)) / true_tr(bi);

            %% 4) NA-Hutch++ (KR)
            t0 = tic;
            opts = struct( ...
              'c1',      1/6, ...
              'c2',      1/3, ...
              'kr_dist', 'gaussian', ...
              'sphere',  true     ...
            );
            Z4 = na_hutchplusplus_KR(Afun, k, n, ell, opts);
            times(bi,run,idx,4) = toc(t0);
            errs(bi,run,idx,4)  = abs(Z4 - true_tr(bi)) / true_tr(bi);

            fprintf(' errs = [%.1e, %.1e, %.1e, %.1e]\n', ...
                    squeeze(errs(bi,run,idx,1:4)));
        end
    end
end

%% Save
save('trace_data_multiple_beta.mat', ...
    'beta_levels','m_vals','errs','times','true_tr','spectral_decay','L');

fprintf('All experiments complete; data saved.\n');
fprintf('Done.\n');


fprintf('All experiments complete; data saved.\n');
function d = tfim_eigs(n,h)
    if mod(n,2) ~= 0
        error('Not implemented for n odd')
    end
    ks = ((-n+1):2:(n-1))*pi/n;
    energies = 2*sqrt(1+h^2+2*h*cos(ks)).';
    ground = -0.5*sum(energies);
    mat = bitand(uint64(0:(2^n-1))',uint64(2.^(0:(n-1)))) ~= 0;
    d = mat(mod(sum(mat,2),2)==0,:)*energies + ground;
    
    ks = ((-n/2):(n/2-1))*2*pi/n;
    energies = 2*sqrt(1+h^2+2*h*cos(ks)).';
    energies(1) = -2*(1+h);
    energies(n/2+1) = 2*(1-h);
    ground = -0.5*sum(energies);
    d = [d;mat(mod(sum(mat,2),2)==1,:)*energies + ground];
end

function H = tfim(n,h)
    N = 2^n;
    rows = zeros(N*(n+1), 1); cols = zeros(N*(n+1), 1);
    vals = zeros(N*(n+1), 1); idx = 1;
    
    for i = uint64(0):uint64(N-1)
        for j = 0:(n-1)
            mask = uint64(2^j);
            rows(idx) = i+1; cols(idx) = bitxor(i,mask)+1;
            vals(idx) = -h; idx = idx+1;
        end
        val = bitxor(bitget(i,1),bitget(i,n));
        for j = 1:(n-1)
            val = val + bitxor(bitget(i,j),bitget(i,j+1));
        end
        rows(idx) = i+1; cols(idx) = i+1;
        vals(idx) = 2*double(val)-n; idx = idx+1;
    end
    H = sparse(rows, cols, vals, N, N);
end

function Z = hutch_KR(n, Afun, m, ell)
    Omega = krSketch(n, m, ell, "gaussian",true);
    Z = product_trace(Omega', Afun(Omega));
end

function [trest] = nystrompp(n, Afun, Omega, Psi)
    Y = Afun(Omega); Z = Afun(Psi);
    %Regularizaton
    nu = sqrt(n)*eps(norm(Y));
    Y_nu = Y + nu*Omega;
    C = chol(Omega'*Y_nu);
    B = Y_nu/C;
    [U,S,~] = svd(B,'econ');
    Lambda = max(0,S^2-nu*eye(size(S)));
    
    tr1 = trace(Lambda);
    tr2 = 2*(product_trace(Psi',Z) - product_trace(Psi',U*(Lambda*(U'*Psi))))/(size(Omega,2)*2);
    trest = tr1 + tr2;
end

function t = product_trace(A, B)
    t = sum(sum(A .* B.', 2));
end
