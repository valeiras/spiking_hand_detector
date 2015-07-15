% Script generating the training set from the segmented sections.
% The format of the training set is the one used by the 
% DeepLearnToolbox (https://github.com/rasmusbergpalm/DeepLearnToolbox)

clear all
close all

% Time interval for the output images
dt = 10000;

width = 60;
height = 60;

pos_folder = './positive';
neg_folder = './negative';

pos_subfolder = dir(pos_folder);
pos_subfolder = setdiff({pos_subfolder.name}, {'.', '..', 'bad_results'});

neg_subfolder = dir(neg_folder);
neg_subfolder = setdiff({neg_subfolder.name}, {'.', '..', 'bad_results'});

block_size = 1000;

% We preallocate memory for the data arrays (optimization):
imgs = zeros(height, width, block_size);
labels = zeros(2, block_size);

nb_blocks = 1;
count = 1;

for ii = 1:length(pos_subfolder)
    pos_file = dir([pos_folder, '/', pos_subfolder{ii}, '/*.mat']);
    pos_file = {pos_file.name};
    
    neg_file = dir([neg_folder, '/', neg_subfolder{ii}, '/*.mat']);
    neg_file = {neg_file.name};
    
    for jj = 1:length(pos_file)
        % We load the positive sample (the name of the TD file will be
        % positive_td)
        load([pos_folder, '/', pos_subfolder{ii}, '/', pos_file{jj}])
        
        % We load the negative sample (the name of the TD file will be
        % negative_td)
        load([neg_folder, '/', neg_subfolder{ii}, '/', neg_file{jj}])
        
        % We make sure that there are no elements out of range
        positive_td.x(find(positive_td.x<1)) = nan(1);
        positive_td.x(find(positive_td.x>width)) = nan(1);
        positive_td.y(find(positive_td.y<1)) = nan(1);
        positive_td.y(find(positive_td.x>height)) = nan(1);
        
        % We make sure that there are no elements out of range
        negative_td.x(find(negative_td.x<1)) = nan(1);
        negative_td.x(find(negative_td.x>width)) = nan(1);
        negative_td.y(find(negative_td.y<1)) = nan(1);
        negative_td.y(find(negative_td.x>height)) = nan(1);
        
        last_kk = 1;
        if(numel(positive_td.ts)>0)
            ts = positive_td.ts(1);
            
            % We process the positive sample
            while(ts + dt < positive_td.ts(end))
                img = 0.5*ones(width, height);
                kk = find(positive_td.ts > ts + dt, 1, 'first');
                
                pos_idx = last_kk -1 + find(positive_td.p(last_kk:kk)==1);
                neg_idx = last_kk -1 + find(positive_td.p(last_kk:kk)==0);
                
                % We convert it to linear indices
                pos_ind = sub2ind([height, width],...
                    positive_td.y(pos_idx), positive_td.x(pos_idx));
                
                neg_ind = sub2ind([height, width],...
                    positive_td.y(neg_idx), positive_td.x(neg_idx));
                
                % We remove the NaN entries
                pos_ind = pos_ind(find(~isnan(pos_ind)));
                neg_ind = neg_ind(find(~isnan(neg_ind)));
                
                % And create the image
                img(pos_ind) = 1;
                img(neg_ind) = 0;
                
                imgs(:, :, count) = img;
                labels(1, count) = 1;
                
                count = count + 1
                if(count > nb_blocks*block_size)
                    imgs = cat(3, imgs, zeros(height, width, block_size));
                    labels = [labels, zeros(2, block_size)];
                    nb_blocks = nb_blocks + 1;
                end
                
                last_kk = kk;
                ts = ts+dt;
            end
        end
        
        if(numel(negative_td.ts)>0)
            last_kk = 1;
            ts = negative_td.ts(1);
            
            % We process the negative sample
            while(ts + dt < negative_td.ts(end))
                img = 0.5*ones(width, height);
                kk = find(negative_td.ts > ts + dt, 1, 'first');
                
                pos_idx = last_kk -1 + find(negative_td.p(last_kk:kk)==1);
                neg_idx = last_kk -1 + find(negative_td.p(last_kk:kk)==0);
                
                % We convert it to linear indices
                pos_ind = sub2ind([height, width],...
                    negative_td.y(pos_idx), negative_td.x(pos_idx));
                
                neg_ind = sub2ind([height, width],...
                    negative_td.y(neg_idx), negative_td.x(neg_idx));
                
                % We remove the NaN entries
                pos_ind = pos_ind(find(~isnan(pos_ind)));
                neg_ind = neg_ind(find(~isnan(neg_ind)));
                
                % And create the image
                img(pos_ind) = 1;
                img(neg_ind) = 0;
                
                imgs(:, :, count) = img;
                labels(2, count) = 1;
                
                count = count+1
                
                if(count>nb_blocks*block_size)
                    imgs = cat(3, imgs, zeros(height, width, block_size));
                    labels = [labels, zeros(2, block_size)];
                    nb_blocks = nb_blocks + 1;
                end
                
                last_kk = kk;
                ts = ts+dt;
            end
        end
    end
end

% We erase the extra elements in the data arrays
if(count<nb_blocks*block_size)
    imgs(:, :, count:end) = [];
    labels(:, count:end) = [];
end

save('training_set', 'imgs', 'labels')