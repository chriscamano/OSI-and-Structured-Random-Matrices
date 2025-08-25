Nx        = 256;        % grid points per dim (even or odd)
L         = 16;         % half-width of domain
dx        = 2*L/(Nx-1);
x         = linspace(-L, L, Nx);
[X,Y]     = meshgrid(x,x);

g         = 10000;      % interaction strength (β₂)
Omega_max = .9;         % rotation speed (just above Ωc)

dtau      = 1e-4;       % imaginary-time Euler step
maxIter   = 30000;      % imag-time iterations
tol       = 1e-6;       % imag-time convergence tolerance

gamma_i   = 1;          % initial anisotropy
gamma_f   = 1.0;        % final isotropic trap

saveEvery = 1;                        
nSnap     = ceil(maxIter / saveEvery);  
Snap      = complex(zeros(Nx^2, nSnap));
snapIdx   = 0;

fprintf('--- GPE Imag-Time Snapshots + POD ---\n');
fprintf('Nx=%d, L=%.1f, dx=%.3f, g=%.0f\n', Nx, L, dx, g);
fprintf('Imag-time: dtau=%.1e, maxIter=%d, tol=%.1e, saveEvery=%d\n\n', dtau, maxIter, tol, saveEvery);
%% ============= Initial State =============
mu    = L^2;
rhoTF = max(mu - 0.5*(X.^2 + Y.^2), 0);
phi   = sqrt(rhoTF) .* (1 + 0.01*(randn(size(X)) + 1i*randn(size(X))));
phi   = phi / sqrt(sum(abs(phi(:)).^2)*dx^2);


%% Precompute spectral operator
dk      = pi / L;
kidx    = ifftshift(-floor(Nx/2):(ceil(Nx/2)-1));
kvec    = dk * kidx;
[KX,KY] = meshgrid(kvec,kvec);
X2     = X.^2;
Y2     = Y.^2;
LapSym = -(KX.^2 + KY.^2);

for it = 1:maxIter
    phi_old = phi;
    
    ph_hat = fft2(phi);
    
    % laplacian and angular‐momentum term
    lapPhi = ifft2(LapSym .* ph_hat);
    dpx     = ifft2(1i * KX .* ph_hat);
    dpy     = ifft2(1i * KY .* ph_hat);
    LzPhi   = -1i * (X .* dpy - Y .* dpx);
    
    frac  = max(0, (maxIter - it)/maxIter);
    gamma = gamma_f + frac*(gamma_i - gamma_f);
    
    % static trap potential plus scaling in Y
    Vtrap = 0.5 * (X2 + (gamma*Y).^2);
    rhs = 0.5*lapPhi - Vtrap.*phi - g.*abs(phi).^2 .* phi + Omega_max .* LzPhi;
    
    %normalization 
    phi = phi + dtau * rhs;
    phi = phi / sqrt(sum(abs(phi(:)).^2) * dx^2);
    
    if mod(it, saveEvery) == 0
        snapIdx      = snapIdx + 1;
        Snap(:,snapIdx) = phi(:);
    end
    
    if mod(it,500) == 0
        dmax = max(abs(phi(:) - phi_old(:)));
        fprintf('  iter %5d/%5d, Δ=%.2e, γ=%.3f\n', it, maxIter, dmax, gamma);
    end
end

Snap = Snap(:,1:snapIdx);
%%
figure('Name','Ground State','Color','w','Units','inches','Position',[2 2 6 5]);
rho_ground = abs(phi).^2;
imagesc(x, x, rho_ground);
axis equal tight; colorbar;
colormap(parula);
title('Ground State Density');
xlabel('x'); ylabel('y');

%% 3) Compute density

% figure('Color','w', 'Renderer','opengl');
% h = surf(X, Y, rho_ground, ...
%     'EdgeColor','none', ...
%     'FaceColor','interp');
% colormap(parula);
% shading interp;                          % smooth color transitions
% axis tight;                              % fit axes to data
% 
% xlabel('X', 'FontSize', 16);             % larger font for labels
% ylabel('Y', 'FontSize', 16);             % larger font for labels
% % zlabel removed
% 
% set(gca, ...
%     'FontSize', 14,'Layer',    'bottom',   'Box',      'on',      'Color',    'none');                 % draw axes/grid beneath surface
%                     % show the surrounding box
%                % transparent background
% 
% view(45, 30);                            % camera angle
% camlight('headlight');                   % add headlight
% lighting('gouraud');                     % smooth lighting
% 
% title('Ground-State Density |\phi|^2', ...
%     'FontSize', 18, 'FontWeight','bold', ...
%     'Interpreter','tex');
% 
% 
% set(gcf, 'PaperPositionMode', 'auto');
% print(gcf, '-dpng', '-r300', 'ground_state_density.png');
% 


%%  POD on imaginary-time snapshots
fprintf('Starting POD\n');
[~, nSnap]   = size(Snap);
SnapReal     = [ real(Snap) ;
                 imag(Snap) ];              % 2*Ngrid × nSnap
meanFieldR   = mean(SnapReal, 2)FieldR * ones(1, nSnap);

fprintf('Running GNSVD (real‐valued POD) on imag-time data...\n');
k      = 1000;
p      = ceil(1.5*k);

Om_SS   = sparseStack(k, size(SnapRealMS,2),4)'; 
Psi_SS  = sparseStack(p, size(SnapRealMS,1),4)'; 
[Ureal, S_pod, V_pod, ~, relErr] = generalized_Nystrom_SVD(SnapRealMS, Om_SS, Psi_SS, true);
fprintf('  POD relErr = %.2e\n\n', relErr);
%%

rows  = size(Ureal,1);
Ngrid = rows/2;
Nx    = sqrt(Ngrid);
modes = 6;;
SnapRealMS   = SnapReal - mea
POD_modes = complex(zeros(Nx, Nx, modes));
for m = 1:modes
    col   = Ureal(:,m);
    phi_m = reshape(col(1:Ngrid) + 1i*col(Ngrid+1:end), Nx, Nx);
    POD_modes(:,:,m) = phi_m;
end
% save(fullfile('pod_modes.mat'),'POD_modes','S_pod');

 
%%
figure('Position',[100 100 1200 600],'Color','w');

for m = 1:modes
    subplot(2,3,m);
    % plot density with a single fast texture
    imagesc(abs(POD_modes(:,:,m)).^2);
    axis image off;
    title(sprintf('POD %d',m));
end
colormap(parula);


