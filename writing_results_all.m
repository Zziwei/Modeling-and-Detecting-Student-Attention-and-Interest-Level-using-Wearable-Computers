% format long;
% time_list = load('monday/non_duplicate_time.txt');
% Files = dir('monday/wifi-data/writing_merged_results_everyone/');
% mixed_data = [];
% for k = 3:length(Files(:))
%     FileNames = Files(k).name;
%     tmp_data = load(strcat('monday/wifi-data/writing_merged_results_everyone/',FileNames));
%     for i = 1:length(tmp_data(:,1))
%         if mod(i,2) == 0
%             tmp_data(i,1) = tmp_data(i,1) - 1;
%         end
%     end
%     mixed_data = [mixed_data; tmp_data];
% end
% 
% output_file = fopen(strcat('monday/wifi-data/writing_all.txt'), 'a');
% mixed_data = sortrows(mixed_data, 2);
% n_writing = 0;
% all_writing = [];
% all_writing = [all_writing; n_writing mixed_data(1,2)];
% for i = 1:length(mixed_data(:,1))
%     if mixed_data(i,1) == 1
%         n_writing = n_writing + 1;
%         all_writing = [all_writing; n_writing mixed_data(i,2)];
%     elseif mixed_data(i,1) == 0
%         n_writing = n_writing - 1;
%         all_writing = [all_writing; n_writing mixed_data(i,2)];
%     end
% end
% 
% plot_mat = [];
% for j = 1:length(all_writing(:,1))
%     if j > 1
%         fprintf(output_file, '%d %ld \r\n', all_writing(j - 1,1), all_writing(j,2));
%         plot_mat = [plot_mat; all_writing(j - 1,1), all_writing(j,2)];
%     end
%     fprintf(output_file, '%d %ld \r\n', all_writing(j,1), all_writing(j,2));
%     plot_mat = [plot_mat; all_writing(j,1), all_writing(j,2)];
% end
% fprintf(output_file, '0 %ld \r\n', time_list(end));
% fclose(output_file);
% 
% dateNumArray  = plot_mat(:,2) ./86400000 + datenum(1970,1,1) - 6/24;
% timeTextArray = datestr(plot_mat(:,2) ./86400000 + datenum(1970,1,1) - 6/24);
% timeArray = timeTextArray(:,13:20);
% count = 1;
% for i = 1:155:size(plot_mat,1)
%     timeReduced(count,:) = timeArray(i,:);
%     count = count + 1;
% end
% 
% hold;
% plot(dateNumArray, plot_mat(:,1));
% for t = 1:length(time_list(:,1))
%     plot([(time_list(t) /86400000 + datenum(1970,1,1) - 6/24) (time_list(t) /86400000 + datenum(1970,1,1) - 6/24)], [0 17],'k');
% end
% set(gca,'XTick',dateNumArray(1:155:size(plot_mat, 1)),'xticklabel',timeReduced);
% % datetick('x',13,'keepticks');



format long;
time_list = load('wednesday/non_duplicate_time.txt');
Files = dir('wednesday/wifi-data/writing_merged_results_everyone/');
mixed_data = [];
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    tmp_data = load(strcat('wednesday/wifi-data/writing_merged_results_everyone/',FileNames));
    for i = 1:length(tmp_data(:,1))
        if mod(i,2) == 0
            tmp_data(i,1) = tmp_data(i,1) - 1;
        end
    end
    mixed_data = [mixed_data; tmp_data];
end

output_file = fopen(strcat('wednesday/wifi-data/writing_all.txt'), 'a');
mixed_data = sortrows(mixed_data, 2);
n_writing = 0;
all_writing = [];
all_writing = [all_writing; n_writing mixed_data(1,2)];
for i = 1:length(mixed_data(:,1))
    if mixed_data(i,1) == 1
        n_writing = n_writing + 1;
        all_writing = [all_writing; n_writing mixed_data(i,2)];
    elseif mixed_data(i,1) == 0
        n_writing = n_writing - 1;
        all_writing = [all_writing; n_writing mixed_data(i,2)];
    end
end

plot_mat = [];
for j = 1:length(all_writing(:,1))
    if j > 1
        fprintf(output_file, '%d %ld \r\n', all_writing(j - 1,1), all_writing(j,2));
        plot_mat = [plot_mat; all_writing(j - 1,1), all_writing(j,2)];
    end
    fprintf(output_file, '%d %ld \r\n', all_writing(j,1), all_writing(j,2));
    plot_mat = [plot_mat; all_writing(j,1), all_writing(j,2)];
end
fprintf(output_file, '0 %ld \r\n', time_list(end));
fclose(output_file);

dateNumArray  = plot_mat(:,2) ./86400000 + datenum(1970,1,1) - 6/24;
timeTextArray = datestr(plot_mat(:,2) ./86400000 + datenum(1970,1,1) - 6/24);
timeArray = timeTextArray(:,13:20);
count = 1;
for i = 1:155:size(plot_mat,1)
    timeReduced(count,:) = timeArray(i,:);
    count = count + 1;
end

hold;
plot(dateNumArray, plot_mat(:,1));
for t = 1:length(time_list(:,1))
    plot([(time_list(t) /86400000 + datenum(1970,1,1) - 6/24) (time_list(t) /86400000 + datenum(1970,1,1) - 6/24)], [0 23],'k');
end
set(gca,'XTick',dateNumArray(1:155:size(plot_mat, 1)),'xticklabel',timeReduced);
% datetick('x',13,'keepticks');