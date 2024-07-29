%% Load data
load('batchInput.mat');
load('phaseFilter.mat');
load('camera_photo.mat');
load('input_image_number.mat');
load('filter_image_number.mat');

%%
cam = single(camera_photo);

%%
max = imregionalmax(cam(:,:,1),8);
[row,col] =  find(max);

%%
h = imhmax(cam(:,:,1), 15);
mesh(h)