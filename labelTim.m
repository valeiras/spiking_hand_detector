% Script for labelling the Tim Cootes dataset (available here: 
% http://personalpages.manchester.ac.uk/staff/timothy.f.cootes/data/) 

clear all
close all

% Folder containing the images
main_folder = './Hand_datasets/Tim_Cootes/';

% We get the list of folders contained in the main one
folders = dir(main_folder);
% And get rid of '.' and '..'
folders = setdiff({folders.name}, {'.','..'})'; 

% Format of the images
img_format = 'jpg';
n = length(img_format) + 1;

for jj=1:size(folders, 1)
    % We get all the images in each folder
    curr_folder = [main_folder, folders{jj}, '/'];
    img_files = dir([curr_folder, '*.', img_format]);
    
    for ii = 1:length(img_files)
        % We load the image
        img = imread([curr_folder, img_files(ii).name]);
        imshow(img)
        set(gcf, 'Pos', [620    42   904   640])
        
        % We apply the conversion
        [xx, yy] = ginput(2);
        
        save([curr_folder, img_files(ii).name(1:end-n)], 'xx', 'yy')
    end
end