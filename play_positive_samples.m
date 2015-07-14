clear all
close all

folder = './positive/';

person = {'andreas', 'cornelia', 'guillaume', 'kostas', 'michael'};
object = {'cup', 'knife', 'spatula', 'sponge', 'spoon'};

for pp = 1:length(person)
    for oo = 1:length(object)
        subfolder = [folder, person{pp}, '_', object{oo}, '/'];
        file = dir([subfolder, '*.mat']);
        file = {file.name};
        
        for ii = 1:length(file)
            load([subfolder, file{ii}])
            player(segmented_td, 10000, false, 'both', 60, 60)
        end
    end
end

close all