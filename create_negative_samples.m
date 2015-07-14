clear all
close all

width = 180;
height = 180;

% Boolean indicating wheter to plot the resutls
plot_res = true;

% Duration of the plotted frames
dt_plot = 10000;
l_2 = 30;
l_over = 20;

% Size of the block for the preallocation of the variables
block_size = 10000;

folder = './recordings_mat/';
person = {'andreas', 'cornelia', 'guillaume', 'kostas', 'michael'};
object = {'cup', 'knife', 'spatula', 'sponge', 'spoon', 'stone'};

% Mean rate in all the segmented data
min_rate = 0.0312;

for pp = 1:length(person)
    for oo = 1:length(object)
        recording = [person{pp}, '_', object{oo}]
        file_nb = 1;
        
        filename = [folder, recording];
        label_file =  ['./labels/', recording, '_label'];
        tracking_file = ['./tracking/', recording, '_tracker'];
        output_folder = ['./negative/', recording];
        mkdir(output_folder)
        
        % We load the test file containing the tracking
        load(tracking_file);

        % We load the file containing the labels
        load(label_file);
        
        % We load the events
        load(filename);
        
        % We get rid of the 10 hottest pixels
        td_data = hide_hot_pixels(td_data, width, height, 10);
        
        % Counter of the tracker array
        count_tracker= 1;
        
        % Last index of the considered segment
        end_jj = idx_start(1) + round(3*(idx_contact(1)-idx_start(1))/4);        
        
        % We get the time interval
        dt = td_data.ts(end_jj) - td_data.ts(idx_start(1));
        
        % We set the minimum number of events
        thresh = min_rate*dt;
        
        ii = 1;
        
        % Numbers of times that we have tried to find a good starting point
        nb_try = 1;
        
        while ii<=length(idx_start)
            last_jj = idx_start(ii);
            last_ts = td_data.ts(last_jj);
            
            % We preallocate memory for the data arrays (optimization):
            negative_td.ts = zeros(1, block_size);
            negative_td.x = zeros(1, block_size);
            negative_td.y = zeros(1, block_size);
            negative_td.p = zeros(1, block_size);
                        
            nb_blocks = 1;
            count_mem = 1;            
            
            count_tracker = 1;
            while(td_data.ts(idx_start(ii)) > ts_tracker(count_tracker) && count_tracker < length(ts_tracker))
                count_tracker = count_tracker +1;
            end
            keeplooping = true;
            
            while(keeplooping)
                % We pick a random point                
                x_cen_neg = randi(width);
                y_cen_neg = randi(height);
                
                % We make sure that it does not coincide with the hand area
                if (abs(x_cen_neg - x_tracker(count_tracker)) < 2*l_over) && (abs(y_cen_neg - y_tracker(count_tracker) < 2*l_over))
                    if abs(x_cen_neg - x_tracker(count_tracker)) > abs(y_cen_neg - y_tracker(count_tracker))
                        s = sign(x_cen_neg - x_tracker(count_tracker));
                        x_cen_neg = x_cen_neg + s*2*l_over  -(x_cen_neg - x_tracker(count_tracker));
                    else
                        s = sign(y_cen_neg - y_tracker(count_tracker));
                        y_cen_neg = y_cen_neg + s*2*l_over  -(y_cen_neg - y_tracker(count_tracker));
                    end
                end
                
                % We count the number of events inside of the box
                ev = (td_data.x(idx_start(ii):end_jj)>= x_cen_neg - l_2).*...
                    (td_data.x(idx_start(ii):end_jj)< x_cen_neg + l_2).*...
                    (td_data.y(idx_start(ii):end_jj)>= y_cen_neg - l_2).*...
                    (td_data.y(idx_start(ii):end_jj)< y_cen_neg + l_2);
                
                nb_ev = sum(ev);
                if(nb_ev >=  thresh)
                    keeplooping = false;
                end
                
                % Every 10 trials we reduce the threshold, so we do not get
                % stacked
                if(mod(nb_try, 10) == 0)
                    thresh = thresh/2;
                    disp('Reducing the threshold')
                end
                nb_try = nb_try + 1;
            end
            
            % We see which events of the segment are inside of the
            % corresponding bounding box
            for jj = idx_start(ii):end_jj                
                % If the current position is later than the tracker, we go to  the
                % next one
                while(td_data.ts(jj) > ts_tracker(count_tracker) && count_tracker < length(ts_tracker))
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
                
                % We make sure that our center does not coincide with the
                % positive example
                if abs(x_cen_neg - x_cen) < 2*l_over && abs(y_cen_neg - y_cen) < 2*l_over
                    if abs(x_cen_neg - x_cen) > abs(y_cen_neg - y_cen)
                        s = sign(x_cen_neg - x_cen);
                        x_cen_neg = x_cen_neg + s*2*l_over - (x_cen_neg - x_cen);
                    else
                        s = sign(y_cen_neg - y_cen);
                        y_cen_neg = y_cen_neg + s*2*l_over - (y_cen_neg - y_cen);
                    end
                end
                
                
                if(td_data.x(jj)>=x_cen_neg-l_2 && td_data.x(jj)<x_cen_neg+l_2...
                        && td_data.y(jj)>=y_cen_neg-l_2 && td_data.y(jj)<y_cen_neg+l_2)
                    
                    negative_td.ts(count_mem) = td_data.ts(jj);
                    negative_td.x(count_mem) = td_data.x(jj) - round((x_cen_neg-l_2));
                    negative_td.y(count_mem) = td_data.y(jj) - round((y_cen_neg-l_2));
                    negative_td.p(count_mem) = td_data.p(jj);
                    
                    count_mem = count_mem+1;
                    
                    % If we have reached the end of the matrices, we
                    % increase their size
                    if(count_mem>nb_blocks*block_size)
                        negative_td.ts = [negative_td.ts, zeros(1, block_size)];
                        negative_td.x = [negative_td.x, zeros(1, block_size)];
                        negative_td.y = [negative_td.y, zeros(1, block_size)];
                        negative_td.p = [negative_td.p, zeros(1, block_size)];
                        
                        nb_blocks = nb_blocks + 1;
                    end
                end
                
                % We plot the results every dt_plot microseconds
                if(plot_res && td_data.ts(jj) - last_ts >= dt_plot)
                    hold off
                    plot(td_data.x(last_jj:jj), td_data.y(last_jj:jj), '.b')
                    hold on
                    
                    xx_neg = [x_cen_neg-l_2, x_cen_neg+l_2, x_cen_neg+l_2, x_cen_neg-l_2, x_cen_neg-l_2];
                    yy_neg = [y_cen_neg-l_2, y_cen_neg-l_2, y_cen_neg+l_2, y_cen_neg+l_2, y_cen_neg-l_2];
                    plot(xx_neg, yy_neg, 'r')
                    
                    xx = [x_cen-l_2, x_cen+l_2, x_cen+l_2, x_cen-l_2, x_cen-l_2];
                    yy = [y_cen-l_2, y_cen-l_2, y_cen+l_2, y_cen+l_2, y_cen-l_2];
                    plot(xx, yy, 'k')
                    
                    axis([0 width 0 height])
                    set(gca, 'YDir', 'reverse')
                    drawnow
                    
                    last_ts = td_data.ts(jj);
                    last_jj = jj;
                end
            end
            
            % We erase the extra elements in the data arrays
            if(count_mem<nb_blocks*block_size)
                negative_td.ts(count_mem:end) = [];
                negative_td.x(count_mem:end) = [];
                negative_td.y(count_mem:end) = [];
                negative_td.p(count_mem:end) = [];
            end
            
            % If there are enough elements, we save the file
            if(length(negative_td.ts)>thresh)
                output_filename = sprintf('%s/negative_%03d', output_folder, file_nb);
                save(output_filename, 'negative_td')
                file_nb = file_nb + 1;
                ii = ii +1;
                
                if ii<= length(idx_start)
                    % We take 3/4 of the inital movement
                    end_jj = idx_start(ii) + round(3*(idx_contact(ii)-idx_start(ii))/4);
                    
                    % We get the time interval
                    dt = td_data.ts(end_jj) - td_data.ts(idx_start(ii));
                    thresh = min_rate*dt;
                    nb_try = 1;
                end
                disp('Found a good starting point, going for the next one')
            else
                disp('Not enough events, we keep looking')
            end
            
        end
        
        close all
    end
end