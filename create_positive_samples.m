clear all
close all

width = 180;
height = 180;

% Duration of the plotted frames
dt_plot = 10000;
l_2 = 30;

% Size of the block for the preallocation of the variables
block_size = 10000;

folder = './recordings_mat/';
person = {'andreas', 'cornelia', 'guillaume', 'kostas', 'michael'};
object = {'cup', 'knife', 'spatula', 'sponge', 'spoon', 'stone'};

for pp = 1:length(person)
    for oo = 1:length(object)
        recording = [person{pp}, '_', object{oo}]
        
        filename = [folder, recording];
        label_file =  ['./labels/', recording, '_label'];
        tracking_file = ['./tracking/', recording, '_tracker'];
        output_folder = ['./positive/', recording, '/'];
        mkdir(output_folder)
        
        % We load the file containing the tracking
        load(tracking_file);
              
        % We load the file containing the labels
        load(label_file);
        
        % We load the events
        load(filename);
        
        % Counter of the tracker position
        count_tracker= 1;
        
        % Number of the file to be written
        file_nb = 1;
        
        for ii=1:length(idx_start)
            disp(['Segment ', num2str(ii)])
            last_jj = idx_start(ii);
            last_ts = td_data.ts(last_jj);
            
            % We preallocate memory for the data arrays (optimization):
            segmented_td.ts = zeros(1, block_size);
            segmented_td.x = zeros(1, block_size);
            segmented_td.y = zeros(1, block_size);
            segmented_td.p = zeros(1, block_size);
            
            nb_blocks = 1;
            
            % Counter of the position in the data matrix
            count_data = 1;
            
            % We take 3/4 of the inital movement
            end_jj = idx_start(ii) + round(3*(idx_contact(ii)-idx_start(ii))/4);
            
            for jj = idx_start(ii):end_jj
                
                % If the current position is later than the tracker, we go to  the
                % next one
                if(td_data.ts(jj) > ts_tracker(count_tracker) && count_tracker < length(ts_tracker))
                    count_tracker = count_tracker +1;
                end
                
                % We interpolate
                if(count_tracker+1<length(ts_tracker))
                    x_cen = x_tracker(count_tracker) + (x_tracker(count_tracker+1)-x_tracker(count_tracker))*...
                        (td_data.ts(jj)-ts_tracker(count_tracker))/(td_data.ts(jj)-ts_tracker(count_tracker));
                    y_cen = y_tracker(count_tracker) + (y_tracker(count_tracker+1)-y_tracker(count_tracker))*...
                        (td_data.ts(jj)-ts_tracker(count_tracker))/(td_data.ts(jj)-ts_tracker(count_tracker));
                else
                    x_cen = x_tracker(count_tracker);
                    y_cen = y_tracker(count_tracker);
                end
                
                % If the event is inside of the bounding box we register it
                if(td_data.x(jj)>=x_cen-l_2 && td_data.x(jj)<x_cen+l_2...
                        && td_data.y(jj)>=y_cen-l_2 && td_data.y(jj)<y_cen+l_2)
                    segmented_td.ts(count_data) = td_data.ts(jj);
                    segmented_td.x(count_data) = td_data.x(jj) - (x_cen-l_2);
                    segmented_td.y(count_data) = td_data.y(jj) - (y_cen-l_2);
                    segmented_td.p(count_data) = td_data.p(jj);
                    
                    count_data = count_data+1;
                    
                    % If we have reached the end of the matrices, we
                    % increase their size
                    if(count_data>nb_blocks*block_size)
                        segmented_td.ts = [segmented_td.ts, zeros(1, block_size)];
                        segmented_td.x = [segmented_td.x, zeros(1, block_size)];
                        segmented_td.y = [segmented_td.y, zeros(1, block_size)];
                        segmented_td.p = [segmented_td.p, zeros(1, block_size)];
                        
                        nb_blocks = nb_blocks + 1;
                    end
                end
                
                % We plot the results every dt_plot microseconds
                if(td_data.ts(jj) - last_ts >= dt_plot)
                    hold off
                    plot(td_data.x(last_jj:jj), td_data.y(last_jj:jj), '.b')
                    hold on
                    xx = [x_cen-l_2, x_cen+l_2, x_cen+l_2, x_cen-l_2, x_cen-l_2];
                    yy = [y_cen-l_2, y_cen-l_2, y_cen+l_2, y_cen+l_2, y_cen-l_2];
                    
                    plot(xx, yy, 'r')
                    axis([0 width 0 height])
                    set(gca, 'YDir', 'reverse')
                    drawnow
                    
                    last_ts = td_data.ts(jj);
                    last_jj = jj;
                end
            end
            
            % We erase the extra elements in the data arrays, and save the
            % file
            if(count_data<nb_blocks*block_size)
                segmented_td.ts(count_data:end) = [];
                segmented_td.x(count_data:end) = [];
                segmented_td.y(count_data:end) = [];
                segmented_td.p(count_data:end) = [];
            end
            output_filename = sprintf('%ssegment_%03d', output_folder, file_nb);
            save(output_filename, 'segmented_td')
            file_nb = file_nb + 1;            
        end       
        close all
    end
end
