function [sel, times, absErr, relErr, names] = suite_sparse_driver(sketchFunc, saveName, k, size_limits, max_matrices)
    % suite_sparse_driver  Run randomized SVD benchmarks over SuiteSparse matrices.
    %   [sel, times, absErr, relErr, names] = suite_sparse_driver(sketchFunc, saveName, k, size_limits, max_matrices)
    %   sketchFunc  - function handle @(d, k) -> Omega sketch of size d×k
    %   saveName    - base string for output .mat file (appends '_SuiteSparse_data.mat')
    %   k           - target rank for SVD
    %   size_limits - two‐element vector [minRows, maxRows]
    %   max_matrices- maximum number of matrices to process

    %========== Draw Suite Sparse Matrices ==========
    idx_all = ssget;

    min_sz = size_limits(1);
    max_sz = size_limits(2);

    mask = [idx_all.nrows] >= min_sz & [idx_all.nrows] <= max_sz ...
         & [idx_all.ncols] >= min_sz & [idx_all.ncols] <= max_sz;

    sel     = find(mask);
    num_all = numel(sel);
    num_mat = min(num_all, max_matrices);

    fprintf('Found %d matrices in range, processing %d.\n', num_all, num_mat);

    times  = zeros(num_mat, 6);
    absErr = zeros(num_mat, 1);
    relErr = zeros(num_mat, 1);
    names  = cell(num_mat, 1);

    %========== Main Script ==========
    for i = 1:num_mat
        Prob    = ssget(sel(i));
        A       = Prob.A;
        names{i}= Prob.name;
        normA2  = norm(A, 'fro')^2;

        Omega = sketchFunc(size(A,2), k);

        tTotal = tic;
            t1 = tic; Y = full(A * Omega);             times(i,1) = toc(t1);
            t2 = tic; [Q,~] = qr(Y, 0);                times(i,2) = toc(t2);
            t3 = tic; B = Q' * A;                      times(i,3) = toc(t3);
            t4 = tic; [Uhat,S,~] = svd(B, 'econ');     times(i,4) = toc(t4);
            t5 = tic; U = Q * Uhat;                    times(i,5) = toc(t5);
        times(i,6) = toc(tTotal);

        s2         = sum(diag(S).^2);
        absErr(i)  = sqrt(max(normA2 - s2, 0));
        relErr(i)  = absErr(i) / sqrt(normA2);

        fprintf('%3d/%3d  t=[%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f]  abs_err=%.3e  rel_err=%.3e  name=%s\n', ...
                i, num_mat, times(i,:), absErr(i), relErr(i), names{i});
    end

    %========== Save Results ==========
    dataDir = 'data/suite_sparse_experiment';
    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end
    fileName = fullfile(dataDir, sprintf('%s_SuiteSparse_data.mat', saveName));
    save(fileName, 'sel', 'k', 'times', 'absErr', 'relErr', 'names');
    fprintf('Results saved to %s\n', fileName);
end