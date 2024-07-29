%% Load data
load('batchInput.mat');
load('phaseFilter.mat');
load('camera_photo.mat');
load('input_image_number.mat');
load('filter_image_number.mat');

%%
cam = single(camera_photo);

%%
% max = imregionalmax(cam(:,:,1),9);
% [row,col] =  find(max);
frame = cam(:,:,1);
max = max(frame(:));

%%
[imax, jmax] = find(frame > 10);




%%
% h = imhmax(cam(:,:,1), 15);
% mesh(h)
imagesc(frame); colorbar
hold on
scatter(jmax, imax, 'ro')

%%
mesh(frame)
hold on
for ii=1:length(imax)
    scatter3(jmax(ii), imax(ii), frame(imax(ii), jmax(ii)))
    hold on
end

%%
reg = ordfilt2(frame, 9, ones(3,3), 'zeros');
imagesc(reg)
