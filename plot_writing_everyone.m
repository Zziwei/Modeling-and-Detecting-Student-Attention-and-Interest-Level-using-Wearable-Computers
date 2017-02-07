Files = dir('wednesday/wifi-data/writing_merged_results_everyone/');
scrsz = get(0,'ScreenSize');
for k = 3:length(Files(:))
    close all;
    FileNames = Files(k).name;
    raw_data = load(strcat('wednesday/wifi-data/writing_merged_results_everyone/',FileNames));
    
    id = strsplit(FileNames,'.');
    id = id(1);
    id_fig = strcat(id, '.fig');
    id_png = strcat(id, '.png');
    h = figure;
    plot(raw_data(:,2),raw_data(:,1));
    set(h,'Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
    title(strcat('Writing Recogntion for ', id));
    ylabel('Writing');

    file_prefix = strcat('wednesday/wifi-data/plot_writing_everyone/', id);
    saveas(h, file_prefix{1},'fig');
    saveas(h, file_prefix{1},'png');
end