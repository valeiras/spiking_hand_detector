% Script taking the Tim Cootes dataset (available here: 
% http://personalpages.manchester.ac.uk/staff/timothy.f.cootes/data/) 
% and converting all of its images to DVS-like output.

clear all
close all

% Output dimensions
output_l = 45;

% Main folder containing the different gestures
main_folder = './Hand_datasets/Tim_Cootes/';

% We get the list of folders contained in the main one
folders = dir(main_folder);
% And get rid of '.' and '..'
folders = setdiff({folders.name}, {'.','..'})'; 

% Output folders
output_img_folder = './hand_kernels/';

% Format of the images
img_format = 'jpg';
output_format = 'bmp';

% We check if the output folder already contains images, to decide the
% numbering of out ouput images
output_files = dir([output_img_folder, '*.', output_format]);
count = length(output_files) + 1;

edge_method = 'Sobel';
edge_thresh = 0.12;

for jj=1:size(folders, 1)
    % For each gesture, we get just the first recording
    curr_folder = [main_folder, folders{jj}, '/'];
    img_files = dir([curr_folder, '*.', img_format]);
    annotation_files = dir([curr_folder, '*.mat']);
    
    for ii = 1:length(img_files)
        % We load the image and convert it to grayscale
        img = imread([curr_folder, img_files(ii).name]);
        img = rgb2gray(img);
                
        % We load the annotation
        load([curr_folder, annotation_files(ii).name])
        
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
        output_filename = sprintf('%simg_%06d.%s', output_img_folder, count, output_format)
        imwrite(dvs_output, output_filename)
        count = count +1;
    end
end