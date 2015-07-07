% Script for labelling the NUS hand gesture dataset (available here: 
% http://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/)

clear all
close all

% Folder containing the images and the annotations
img_folder = '.(Hand_datasets/NUS Hand Posture Dataset/Color/';

% Format of the images
img_format = 'jpg';
n = length(img_format) + 1;

% Output folders
output_folder = './Hand_datasets/NUS Hand Posture Dataset/annotations/';

% We scan the folders containing the images and the annotations
img_files = dir([img_folder, '*.', img_format]);

for ii = 121:length(img_files)   
    % We load the image
    img = imread([img_folder, img_files(ii).name]);
    imshow(img)
    set(gcf, 'Pos', [620    42   904   640])
    
    % We apply the conversion
    [xx, yy] = ginput(2);
    
    save([output_folder, img_files(ii).name(1:end-n)], 'xx', 'yy')
end