%% The images for the correlation are obtained from 0-300_10x_100Hz_45um_frame_stack_every10um.avi
% Images
% folder = 'E:\C\ims';
% filenames = dir(fullfile(folder, '*.png'));
% total = numel(filenames);
% 
% for n=1:total
%     f = fullfile(folder, filenames(n).name);
%     images(:, :, n) = imread(f);
% end

for i=1:31
    images(:, :, i) = x0_300_10x_100Hz_45um_frame_stack_every10um(i).cdata(:, :, 1);
end

%% LUT
folder = 'E:\C\LUT_MANUAL';
filenames = dir(fullfile(folder, '*.png'));
total = numel(filenames);

for n=1:total
    f = fullfile(folder, filenames(n).name);
    lut_images(:, :, n) = imread(f);
end

%%
k=11;
subplot(1,2,1)
imshow(images(:, :, k))
colormap gray
axis square

subplot(1,2,2)
imshow(lut_images(:, :, k))
colormap gray
axis square

%%
im_bin = zeros(size(images));
im_bin(images >= mean(images(:))) = 255;

lut_bin = zeros(size(lut_images));
lut_bin(lut_images >= mean(images(:))) = 255;

%%
k=10;
subplot(1,2,1)
imshow(im_bin(:, :, k))
colormap gray
axis square

subplot(1,2,2)
imshow(lut_bin(:, :, k))
colormap gray
axis square












