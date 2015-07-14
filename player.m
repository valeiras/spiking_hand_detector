function player(td_data, dt, pause, type_ev, width, height)
%PLAYER(td_data, dt, pause = false, type_ev = 'both', width = 240, height = 304, pause = false)
% Displays a recording, given by td_data. Frames are constructed by 
% plotting together events occurring inside of a time window of length dt.
%
%   Parameters;
%    - td_data: structure containing the coordinates, timestamp and
%               polarity of the events. It can be obtained using 
%               load_atis_data.
%    - dt: length of the time window.
%    - pause: boolean indicating if we should wait for a button press after
%             each frame.
%    - type_ev: string indicating the type of events to be displayed.
%               Possible valeues are 'on', 'off' and 'both'.
%    - width, height: dimensions of the image plane. They default to atis
%      dimensions

if ~exist('type_ev','var')
    type_ev = 'both';
end

if ~exist('width','var')
    width = 304;
end

if ~exist('height','var')
    height = 240;
end

if ~exist('pause','var')
    pause = false;
end

plot_on = true;
plot_off = true;

if(strcmp(type_ev, 'on'))
    plot_off = false;
elseif(strcmp(type_ev, 'off'))
    plot_on = false;
elseif(~strcmp(type_ev, 'both'))
    error('Incorrect value for type_ev: possible values are ''on'', ''off'' or ''both''.');
end
    
t = td_data.ts(1);
last_ii = 1;

while(t+dt<td_data.ts(end))
    % We find the next index
    ii = find(td_data.ts>=t+dt, 1, 'first');
    % We find the on and off events
    on_ev = (td_data.p(last_ii:ii) == 1);
    off_ev = 1-on_ev;
    hold off
    % /!\ This way of plotting the events will result in events
    % continuously plotted in [0, 0], but the player seems to be much
    % faster than when using find
    if(plot_on)
        plot(td_data.x(last_ii:ii).*on_ev, td_data.y(last_ii:ii).*on_ev, '.b')
        hold on
    end
    if(plot_off)
        plot(td_data.x(last_ii:ii).*off_ev, td_data.y(last_ii:ii).*off_ev, '.r')
    end
    axis([0 width 0 height])
    set(gca, 'YDir', 'reverse')
    title(['t = ', num2str(t)])
    drawnow
    if pause
        waitforbuttonpress
    end
    t = t+dt;
    last_ii = ii;
end

end

