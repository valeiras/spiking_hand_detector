% Script taking the NUS hand gesture dataset (available here: 
% http://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/)) and converting all 
% the images to DVS-like output.
%
% Note: the annotations are not part of the dataset, and they were manually
% obtained

clear all
close all

% Output dimensions
output_l = 45;

% Folders containing the images and the annotations
img_folder = './Hand_datasets/NUS Hand Posture Dataset/BW/';
annotation_folder = './Hand_datasets/NUS Hand Posture Dataset/annotations/';

% Output folders
output_img_folder = './hand_kernels/';

% Format of the images
img_format = 'jpg';
output_format = 'bmp';

% We scan the folders containing the images and the annotations
img_files = dir([img_folder, '*.', img_format]);
annotation_files = dir([annotation_folder, '*.mat']);

% We check that there is one annotation for every image
if(length(img_files)~=length(annotation_files))
    error('The number of annotations and images is not the same!!');
end

% We check if the output folder already contains images, to decide the
% numbering of out ouput images
output_files = dir([output_img_folder, '*.', output_format]);
count = length(output_files) + 1;

edge_method = 'Sobel';
edge_thresh = 0.08;

for ii = 1:length(img_files)   
    % We load the image
    img = imread([img_folder, img_files(ii).name]);
    
    % We load the annotation
    load([annotation_folder, annotation_files(ii).name])
    % We round the coordinates of the bounding box, and make sure that they
    % are inside of the image
    xx = max(1, round(xx));
    xx = min(xx, size(img, 2));
    yy = max(1, round(yy));
    yy = min(yy, size(img, 1));
    
    % We take the corresponding region
    img = img(yy(1):yy(2), xx(1):xx(2));
    
    % We apply the conversion
    dvs_output = img2dvs(img, output_l, edge_method, edge_thresh, false);
    
    % And save the file
    output_filename = sprintf('%simg_%06d.%s', output_img_folder, count, output_format);
    imwrite(dvs_output, output_filename)
    count = count +1;
end