nCores = feature('numcores');
%========== Parameters ==========
k            = 200;                % target rank
size_limits  = [300, 500000];      % 1k ≤ rows and cols ≤ 50k
max_matrices = 3000;                 % maximum problems
zeta         = 4;                  % expected nnz per column

%========== Sketch samplers ==========
gaussSampler            = @(d,k) randn(d, k);
sparseSignSubColSampler = @(d,k) sparseStack(k, d, zeta)';
krComplexSphere_log2d   = @(d,k) krSketch(d, k, ceil(log2(d)), 'cgaussian',  true);
SRTTSampler             = @(d,k) srtt_explicit(d, k, 1, 'dct', 'sparsecol', zeta);

%========== Run benchmarks ==========
lb = size_limits(1);
ub = size_limits(2);

[selG,             timesG,             absErrG,             relErrG] = ...
    suite_sparse_driver(gaussSampler,            'Gauss',             k, size_limits, max_matrices);

[selSsc,           timesSsc,           absErrSsc,           relErrSsc] = ...
    suite_sparse_driver(sparseSignSubColSampler, 'SparseSignIsubcols', k, size_limits, max_matrices);

[selSRTT,          timesSRTT,          absErrSRTT,          relErrSRTT] = ...
    suite_sparse_driver(SRTTSampler,             'SRRT',              k, size_limits, max_matrices);

[selKRcomplexSphereL, timesKRcomplexSphereL, absErrKRcomplexSphereL, relErrKRcomplexSphereL] = ...
    suite_sparse_driver(krComplexSphere_log2d,   'KRcomplexsphere',   k, size_limits, max_matrices);

