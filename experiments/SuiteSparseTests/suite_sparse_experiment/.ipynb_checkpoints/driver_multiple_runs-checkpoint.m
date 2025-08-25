%==========================================================================
% suite_sparse_driver_batch.m
%==========================================================================
nCores = feature('numcores');

%========== Parameters ==========
k            = 200;                % target rank
size_limits  = [300, 500000];      % 1k ≤ rows and cols ≤ 50k
max_matrices = 3000;               % maximum problems
zeta         = 4;                  % expected nnz per column
nTrials      = 3;                  % number of repetitions

%========== Sketch samplers ==========
gaussSampler            = @(d,k) randn(d, k);
sparseSignSubColSampler = @(d,k) sparseStack(k, d, zeta)';
krComplexSphere_log2d   = @(d,k) krSketch(d, k, ceil(log2(d)), 'cgaussian',  true);
SRTTSampler             = @(d,k) srtt_explicit(d, k, 1, 'dct', 'sparsecol', zeta);

samplers = { ...
    struct('fn',gaussSampler,            'name','Gauss'), ...
    struct('fn',sparseSignSubColSampler,'name','SparseSignIsubcols'), ...
    struct('fn',SRTTSampler,            'name','SRRT'), ...
    struct('fn',krComplexSphere_log2d,  'name','KRcomplexsphere')  ...
};

%========== Run benchmarks in batch ==========
results(nTrials) = struct();  % pre-allocate struct array

for t = 1:nTrials
    fprintf('=== Trial %d of %d ===\n', t, nTrials);
    for s = 1:numel(samplers)
        sampler = samplers{s};
        [sel, times, absErr, relErr] = ...
            suite_sparse_driver( sampler.fn, sampler.name, ...
                                 k, size_limits, max_matrices );
        % store outputs
        results(t).(sampler.name).sel    = sel;
        results(t).(sampler.name).times  = times;
        results(t).(sampler.name).absErr = absErr;
        results(t).(sampler.name).relErr = relErr;
    end
end

%========== Save all results ==========
save('suite_sparse_benchmarks_5trials.mat','results','-v7.3');