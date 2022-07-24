% 2048C, pure MPI
% timings: [matmul, copy A & B, reduce-scatter C, other]

% COSMA, library selected optimal strategy
cosma_square = [1.95, 0.37, 0.20, 2.65-1.95-0.37-0.20];
cosma_largeK = [0.69, 0.04, 0.10, 0.84-0.69-0.04-0.11];
cosma_largeM = [0.69, 0.08, 0.04, 0.82-0.69-0.08-0.04];
cosma_flat   = [0.76, 0.19, 0.06, 1.03-0.76-0.19-0.06];

% CA3DMM 
ca3dmm_square = [1.92, 0.20, 0.30, 2.46-1.92-0.20-0.32];
ca3dmm_largeK = [0.67, 0.04, 0.07, 0.78-0.67-0.04-0.07];
ca3dmm_largeM = [0.66, 0.08, 0.08, 0.82-0.66-0.08-0.08];
ca3dmm_flat   = [0.75, 0.14, 0.13, 1.02-0.75-0.14-0.13];

font_size = 16;
fig1 = figure('Renderer', 'painters', 'Position', [10 10 800 600]);
X = categorical({'COSMA','CA3DMM'});
X = reordercats(X, {'COSMA','CA3DMM'});

%%
subplot(1, 5, 1);
square_times = [cosma_square; ca3dmm_square];
square_times = square_times ./ sum(cosma_square);
bar(X, square_times, 'stacked');
ylabel('Relative Runtime (COSMA sum = 1)', 'FontSize', font_size);
yticks(0 : 0.1 : 1);
title('square', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot(1, 5, 2);
largeK_times = [cosma_largeK; ca3dmm_largeK];
largeK_times = largeK_times ./ sum(cosma_largeK);
bar(X, largeK_times, 'stacked');
yticks(0 : 0.1 : 1);
title('largeK', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot(1, 5, 3);
largeM_times = [cosma_largeM; ca3dmm_largeM];
largeM_times = largeM_times ./ sum(cosma_largeM);
bar(X, largeM_times, 'stacked');
yticks(0 : 0.1 : 1);
title('largeM', 'FontSize', font_size);
set(gca, 'FontSize', font_size);

%%
subplot(1, 5, 4);
flat_times = [cosma_flat; ca3dmm_flat];
flat_times = flat_times ./ sum(cosma_flat);
bar(X, flat_times, 'stacked');
legend({'compute', 'replicate $A$, $B$', 'reduce $C$', 'other'}, ...
       'interpreter', 'latex', 'FontSize', font_size);
yticks(0 : 0.1 : 1);
title('flat', 'FontSize', font_size);
set(gca, 'FontSize', font_size);