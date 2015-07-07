function [output_imgs] = imrotate_and_translate(input_img, extra_px, trans_step, angle_min, angle_max, angle_step)
%IMROTATE_AND_TRANSLATE Rotates and translates an image, yielding a new set
%   output_imgs = IMROTATE_AND_TRANSLATE(input_img, extra_px, trans_step, angle_min, angle_max, angle_step)
%   gets an image input_img and translates and rotates it, generating a
%   cell of images output_imgs. 
%
%   Parameters:
%       - extra_px: the output images will have this extra number of pixels
%                   at each side
%       - trans_step: translation step
%       - angle_min, angle_max: minimum and maximum angle for the rotation
%       - angle_step: rotation step

% We get the size of the input image
h = size(input_img, 1);
w = size(input_img, 2);
c = size(input_img, 3);

% And compute the size of the ouput images
new_h = h + 2*extra_px;
new_w = w + 2*extra_px;

% We compute the number of output images
nb_rot = round((angle_max - angle_min)/angle_step) + 1;
nb_trans = round(2*extra_px/trans_step) + 1;
nb_imgs = nb_rot*nb_trans*nb_trans;

% And preallocate memory for them
empty_img = zeros(new_h, new_w, c);
output_imgs = repmat({empty_img}, nb_imgs, 1);

count = 1;
for angle = angle_min:angle_step:angle_max
    % We rotate the image
    rotated_img = imrotate(input_img, angle, 'bicubic', 'crop');
    for x_trans = 1:trans_step:2*extra_px+1
        for y_trans = 1:trans_step:2*extra_px+1
            % And translate it in the y and x axis
            output_imgs{count}(y_trans:y_trans+h-1, x_trans:x_trans+w-1, :) = rotated_img;
            count = count +1;
        end
    end
end

end

