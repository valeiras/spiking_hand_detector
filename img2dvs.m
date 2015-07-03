clear all
close all

% Output dimensions
output_h = 128;
output_w = 128;
    
output_l = 32;

output_idx = 1;

% We preallocate the memory
temp = zeros(4, 1);
max_nb_hands = 100;
xx = repmat({temp}, max_nb_hands);
yy = repmat({temp}, max_nb_hands);
   
h1 = figure;

for img_idx = 1:100;
    figure(h1)
    % We load the images and the annotations
    img_filename = sprintf('hand_dataset/training_dataset/training_data/images/VOC2010_%d.jpg', img_idx);
    anotation = sprintf('hand_dataset/training_dataset/training_data/annotations/VOC2010_%d', img_idx);
    
    load(anotation)
    img = imread(img_filename);
    gray_img = rgb2gray(img);
        
    subplot(2, 3, 1)
    hold off
    imshow(gray_img)
    hold on
    
    % Input dimemsions
    [input_h, input_w, ~] = size(img);
    
    % We arrange the bounding boxes in vectors and plot them
    for ii=1:length(boxes)
        xx{ii} = [boxes{ii}.a(2), boxes{ii}.b(2), boxes{ii}.c(2), boxes{ii}.d(2)];
        yy{ii} = [boxes{ii}.a(1), boxes{ii}.b(1), boxes{ii}.c(1), boxes{ii}.d(1)];        
        plot(xx{ii}([1:4, 1]), yy{ii}([1:4, 1]), 'r')
    end
    axis equal
    
    for ii=1:length(boxes)
        output_filename = sprintf('hand_dataset/training_dataset/training_data/treated_images/img_%06d.jpg', output_idx);
        cropped_output_filename = sprintf('hand_dataset/training_dataset/training_data/treated_images/img_cropped_%06d.jpg', output_idx);
        output_idx = output_idx + 1;
        
        output_img = false(output_h, output_w);
        cropped_output_img = false(output_h, output_w);
        
        subplot(2, 3, 4)
        
        % We get just the correponding region
        % Some points can be out of the image, so we make sure that this does not happen
        x_min = max(floor(min(xx{ii})), 1);
        x_max = min(ceil(max(xx{ii})), size(img, 2));
        y_min = max(floor(min(yy{ii})), 1);
        y_max = min(ceil(max(yy{ii})), size(img, 1));
        
        % We shift the values of the limits, in order to get them in the
        % new image
        xx_shift = xx{ii}-x_min+1;
        yy_shift = yy{ii}-y_min+1;
        
        % We get the corresponding area
        hand_img = gray_img(y_min:y_max, x_min:x_max);
        % We normalize the image
        hand_img = normalize_image(hand_img);
        
        hold off
        imshow(hand_img)
        hold on
        plot(xx_shift([1:4, 1]), yy_shift([1:4, 1]), 'r')
        axis equal
        
        subplot(2, 3, 2)
        % We extract the edges
        edges_hand = edge(hand_img, 'Sobel', 0.05);
         
        hold off
        imshow(edges_hand)
        hold on
        plot(xx_shift([1:4, 1]), yy_shift([1:4, 1]), 'r')
        axis equal
        
        subplot(2, 3, 5)
        % We create the mask (the output will be logical)
        mask = poly2mask(xx_shift, yy_shift, size(hand_img, 1), size(hand_img, 2));
        % We apply the mask to each one of the channels
        cropped_edges_hand = edges_hand.*mask;
        
        imshow(cropped_edges_hand)
        axis equal
        
        subplot(2, 3, 3)
        
        % We make the edges image square, adding zeros all around
        %edges_hand = uint8(edges_hand);
        %cropped_edges_hand = uint8(cropped_edges_hand);
        
        [h, w] = size(edges_hand);
        edges_hand_sq = false(max(h, w));
        cropped_edges_hand_sq = false(max(h, w));
        
        if(h>w)
            st = ceil((h-w)/2);
            edges_hand_sq(:, st:st+w-1) = edges_hand;
            cropped_edges_hand_sq(:, st:st+w-1) = cropped_edges_hand;
        elseif(w>h)
            st = ceil((w-h)/2);
            edges_hand_sq(st:st+h-1, :) = edges_hand;
            cropped_edges_hand_sq(st:st+h-1, :) = cropped_edges_hand;
        else
            edges_hand_sq = edges_hand;
            cropped_edges_hand_sq = cropped_edges_hand;
        end
        
        % Then we resize it with the desired length
        edges_hand_sq = imresize(edges_hand_sq, [output_l, output_l]);
        cropped_edges_hand_sq = imresize(cropped_edges_hand_sq, [output_l, output_l]);
        
        % And we include it in the image, in the corresponding position
        x_min_output = round(x_min*output_w/input_w);
        y_min_output = round(y_min*output_w/input_w);
        
        % Because of the change of scale, we need to make sure that we are
        % in the good range
        x_min_output = max(1, x_min_output);
        x_min_output = min(x_min_output, output_w-output_l+1);
        y_min_output = max(1, y_min_output);
        y_min_output = min(y_min_output, output_h-output_l+1);
        
        output_img(y_min_output:y_min_output+output_l-1, x_min_output:x_min_output+output_l-1) = edges_hand_sq;
        cropped_output_img(y_min_output:y_min_output+output_l-1, x_min_output:x_min_output+output_l-1) = cropped_edges_hand_sq;
        
        imshow(output_img)
        axis equal
        
        subplot(2, 3, 6)
        imshow(cropped_output_img)
        axis equal
        
        imwrite(output_img, output_filename)
        imwrite(cropped_output_img, cropped_output_filename)
        
        waitforbuttonpress
    end
end