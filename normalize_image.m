function [output_img] = normalize_image(input_img, new_min, new_max)
%NORMALIZE_IMGE Normalizes an image, setting its new min and max
%   If the new range is not specified, it defaults to the maximum range
%   possible for the type of image.

if(nargin<3)
    range = getrangefromclass(input_img);
    new_min = range(1);
    new_max = range(2);
end

% We retrieve the type of the original image
img_type = class(input_img);

% We convert the image to double, in order to correctly perform
% mathematical operations with it
input_img = double(input_img);
output_img = (input_img - min(input_img(:)))*(new_max - new_min)/(max(input_img(:)) - min(input_img(:))) + new_min;

% We give the same type to the output image
output_img = cast(output_img, img_type);

end

