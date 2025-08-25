clear; clc;

% parameters
d           = 50000;                 % ambient dimension
rng(0);
disp("Generating test matrix A...");
A = randn(d);  
%%
k_values    = [500, 2500, 25000];      % sketch widths
num_k       = numel(k_values);
zeta_vals   = 1:1:16;                % zeta values
numZeta     = numel(zeta_vals);
num_runs    = 10;                     % repetitions

% preallocate arrays
times_sparse = zeros(num_runs, numZeta, num_k);
times_gauss  = zeros(num_runs, numZeta, num_k);

% Optional warm–up (uncomment if you see first-iteration jitters)
a= A * sparse_sign_isubcols(k_values(1), d, zeta_vals(1))';
a= A * randn(d, k_values(1));


for ik = 1:num_k
    k = k_values(ik);
    fprintf('=== Testing k = %d ===\n', k);
    for run = 1:num_runs
        fprintf('Run %d of %d...\n', run, num_runs);

        % --- Gaussian baseline: run ONCE per (k, run) ---
        Om_g = randn(k, d);
        t0 = tic;
        Y_g = Om_g * A;
        t_gauss = toc(t0);
        times_gauss(run, :, ik) = t_gauss;  %Constant 

        % --- Sparse-sign: run for EACH zeta ---
        for iz = 1:numZeta
            zeta = zeta_vals(iz);
            Om_s = sparseStack(k, d, zeta);
            t0 = tic;
            Y_s = Om_s * A;
            times_sparse(run, iz, ik) = toc(t0);
        end
    end
end


%% save timing data
raw_data.times_sparse = times_sparse;
raw_data.times_gauss  = times_gauss;
raw_data.zeta_vals    = zeta_vals;
raw_data.k_values     = k_values;
raw_data.num_runs     = num_runs;
raw_data.d            = d;
save('Sparse_sign_timing_multiK.mat', 'raw_data');

%% plot average runtimes for each k
figure('Units','inches','Position',[1 1 12 4],'Renderer','opengl');
tiledlayout(1, num_k, "TileSpacing", "compact", "Padding", "compact");

colors = struct(...
    'expl', [0.8500, 0.3250, 0.0980], ... % orange
    'gaus', [0, 0.4470, 0.7410]);         % blue

axh = gobjects(1,num_k);  % keep axes handles

for ik = 1:num_k
    mean_expl = mean(times_sparse(:,:,ik), 1);
    mean_gaus = mean(times_gauss(:,:,ik),  1);

    axh(ik) = nexttile;   % store handle
    semilogy(zeta_vals, mean_expl, '-o', 'LineWidth', 2.0, ...
             'MarkerSize', 5, 'DisplayName', 'Sparse-sign', ...
             'Color', colors.expl);
    hold on;
    semilogy(zeta_vals, mean_gaus, '-s', 'LineWidth', 2.0, ...
             'MarkerSize', 5, 'DisplayName', 'Gaussian', ...
             'Color', colors.gaus);
    hold off;

    grid on;
    axh(ik).GridAlpha = 0.3;
    axh(ik).YMinorGrid = 'on';
    xlabel('\zeta', 'FontSize', 13);
    ylabel('Avg time (s)', 'FontSize', 13);
    title(sprintf('d = %d, k = %d', d, k_values(ik)), 'FontSize', 14);
    legend('Location', 'northeast', 'FontSize', 10, 'Box', 'off');
    set(axh(ik), 'FontSize', 12, 'LineWidth', 1.0);
end

yl = ylim(axh(end));
for ik = 1:num_k-1
    ylim(axh(ik), yl);
end

print(gcf, 'Sparse_sign_timing_multiK_plot.png', '-dpng', '-r300');
