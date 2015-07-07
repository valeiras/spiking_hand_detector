function [dvs_output] = img2dvs(img_input, output_l, edge_method, edge_thresh, plot_output)
%IMG2DVS Converts a given image to a new one containing DVS-like output
%   dvs_output = IMG2DVS(img_input, output_l, edge_method, edge_thresh)
%   converts an image to a new one, containing DVS-like output.
%   
%   Parameters:
%
%       - img_input: input grayscale image
%       - output_l: side of the output image (in pixels) that will be
%                   square
%       - edge_method: method used for the extraction of the edges
%       - edge_thresh: threshold for the edge extraction process
%       - plot_output: boolean value indicating whether to plot the output


% Scaling to be applies to the input image (that does not have to be
% square). We make the biggest side equal to output_l
scale = output_l/max(size(img_input));

% We make the scaling of the image first, and normalize it
scale_img = imnormalize(imresize(img_input, scale, 'bicubic'));

% And then extract the edges
scale_edges = edge(scale_img, edge_method, edge_thresh);

% We center the edges in our output image, and make the rest of it black
% (it is not guaranteed that the input images are square)
dvs_output = false(output_l);

[h, w] = size(scale_edges);

r0 = round((output_l - h)/2) + 1;
c0 = round((output_l - w)/2) + 1;

dvs_output(r0:r0+h-1, c0:c0+w-1) = scale_edges;

% Finally, we make it a binary image
%dvs_output = dvs_output > 0;

if(plot_output)
    subplot(1, 3, 1)
    imshow(img_input)
        
    subplot(1, 3, 2)
    imshow(scale_img)

    subplot(1, 3, 3)
    imshow(dvs_output)

    waitforbuttonpress
end
