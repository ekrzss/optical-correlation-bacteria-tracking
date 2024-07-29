%% Load data
load('batchInput.mat');
load('phaseFilter.mat');
load('camera_photo.mat');
load('input_image_number.mat');
load('filter_image_number.mat');

%%
% cam = single(camera_photo);
cam =  camera_photo;
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

%%
mesh(camera_photo(:,:,5)); colormap;
xlabel('x', 'FontSize',20); ylabel('y', 'FontSize', 20); title('Correlation', 'FontSize', 25)


%%
plot((20:10:190), (20:10:190), 'LineWidth',3)
hold on
plot((20:10:190), (20:10:190), '.', 'MarkerSize',40)
xlabel('ZPlate (\mu m)', 'FontSize',20); ylabel('DZ (\mu m)', 'FontSize', 20); title('Ground Truth: Particle 2', 'FontSize', 25)
legend('Stage','Propagator', 'Location','nw', 'FontSize', 20)
grid on