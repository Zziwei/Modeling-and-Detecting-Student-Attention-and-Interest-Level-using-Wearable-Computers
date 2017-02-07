Files = dir('monday/wifi-data/motion_level_everyone/');
scrsz = get(0,'ScreenSize');
for k = 3:length(Files(:))
    close all;
    FileNames = Files(k).name;
    raw_data = load(strcat('monday/wifi-data/motion_level_everyone/',FileNames));
    timeTextArray = datestr(raw_data(:,3) ./86400000 + datenum(1970,1,1) - 7/24);
    timeArray = timeTextArray(:,13:20);
    count = 1;
    for i = 1:100:size(raw_data,1)
        timeReduced(count,:) = timeArray(i,:);
        count = count + 1;
    end
    
    id = strsplit(FileNames,'.');
    id = id(1);
    id_fig = strcat(id, '.fig');
    id_png = strcat(id, '.png');
    h(1) = figure;
    plot(1:size(raw_data,1),raw_data(:,1));
    set(h(1),'Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
    set(gca,'XTick',1:100:size(raw_data, 1),'xticklabel',timeReduced);
    title(strcat('Motion Level of Acc for ', id));
    ylabel('Motion Level');
    
    h(2) = figure;
    plot(1:size(raw_data,1),raw_data(:,2));
    set(h(2),'Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
    set(gca,'XTick',1:100:size(raw_data, 1),'xticklabel',timeReduced)
    title(strcat('Motion Level of gyroscope for ', id));
    ylabel('Motion Level');
%     
%     acc_fig = strcat('acc_', id_fig);
%     acc_png = strcat('plot_entropy_everyone/acc_', id);
    acc_id = strcat('monday/wifi-data/plot_motion_level_everyone/acc_', id);
    gyr_id = strcat('monday/wifi-data/plot_motion_level_everyone/gyr_', id);
    saveas(h(1), acc_id{1},'fig');
    saveas(h(1), acc_id{1},'png');
    saveas(h(2), gyr_id{1},'fig');
    saveas(h(2), gyr_id{1},'png');
end