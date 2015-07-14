% This script takes a series of recordings in DVS data format
% and saves them as .mat files.

clear all
close all

folder = '~/recordings/Hands_telluride/dvsdata/';
output_mat_folder = '~/recordings/Hands_telluride/mat/';
files = dir([folder, '*.aedat']);

width = 180;
height = 180;

for ii = 1:length(files)
    filename = [folder, files(ii).name];
    [td_data.x, td_data.y, td_data.p, td_data.ts] = getDVSeventsDavis(filename, 1000e6);
    
    % We flip the y axis
    td_data.y = height - td_data.y;

    % We make the timestamps 64 bits
    td_data.ts = uint64(td_data.ts);
    
    % We verify if there is overflow
    dt = int64(td_data.ts(2:end)) - int64(td_data.ts(1:end-1));
    if(min(dt) < 0)
       jj = find(dt < 0, 1, 'first');
       td_data.ts(jj:end) = uint64(td_data.ts(jj:end)) + 2^32;
    end
    
    % And make the initial timestamp zero
    td_data.ts = td_data.ts - td_data.ts(1);
    
    save([output_mat_folder, files(ii).name(1:end-6)], 'td_data');
end