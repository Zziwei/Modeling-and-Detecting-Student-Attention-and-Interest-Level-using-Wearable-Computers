interval = 30000;
Files = dir('monday/wifi-data/writing_merged_results_everyone/');
mkdir('monday/wifi-data/writing_count_everyone/');
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    output_file = fopen(strcat('monday/wifi-data/writing_count_everyone/', FileNames), 'a');
    raw_data_mat = load(strcat('monday/wifi-data/writing_merged_results_everyone/',FileNames));
    start_time = raw_data_mat(1,2);
    writing_time_index = 2;
    for i = start_time + interval:interval:raw_data_mat(end,2)
        writing_time = 0;
        for j = writing_time_index:2:length(raw_data_mat(:,1))
            if raw_data_mat(j,2) > i
                if raw_data_mat(j,1) == 1
                    writing_time = writing_time + (i - max(raw_data_mat(j - 1,2),i - interval));
                end
                break;
            end
            if raw_data_mat(j,1) == 1
                if j == writing_time_index
                    writing_time = writing_time + (raw_data_mat(j,2) - i + interval);
                else
                    writing_time = writing_time + (raw_data_mat(j,2) - raw_data_mat(j - 1,2));
                end
            end
        end
        writing_time_index = j;
        fprintf(output_file, '%d %ld %ld\r\n', writing_time, i - interval, i);
    end
    writing_time = 0;
    if raw_data_mat(writing_time_index,1) == 1
        writing_time = raw_data_mat(writing_time_index,2) - i;
    end
    for j = writing_time_index:2:length(raw_data_mat(:,1))
        if raw_data_mat(j,1) == 1
            writing_time = writing_time + (raw_data_mat(j,2) - raw_data_mat(j - 1,2)); 
        end
        fprintf(output_file, '%d %ld %ld\r\n', writing_time, i, raw_data_mat(end,2));
    end
    fclose(output_file);
end