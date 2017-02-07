time_list = load('wednesday/non_duplicate_time.txt');
Files = dir('wednesday/wifi-data/writing_merged_results_everyone/');
mkdir('wednesday/wifi-data/writing_rate_everyone/');
slide_writing_file = fopen('slide_writing_rate_list.txt', 'a');
slide_writing_mat = [];
for k = 3:length(Files(:))
    rate_list = zeros(1,length(time_list) - 1);
    FileNames = Files(k).name;
    output_file = fopen(strcat('wednesday/wifi-data/writing_rate_everyone/', FileNames), 'a');
    raw_data_mat = load(strcat('wednesday/wifi-data/writing_merged_results_everyone/',FileNames));
    writing_time_index = 2;
    for i = 2:length(time_list)
        writing_time = 0;
        for j = writing_time_index:2:length(raw_data_mat(:,1))
            if raw_data_mat(j,2) > time_list(i)
                if raw_data_mat(j,1) == 1
                    writing_time = writing_time + (time_list(i) - max(raw_data_mat(j - 1,2), time_list(i - 1)));
                end
                break;
            end
            if raw_data_mat(j,1) == 1
                if j == writing_time_index
                    writing_time = writing_time + (raw_data_mat(j,2) - time_list(i - 1));
                else
                    writing_time = writing_time + (raw_data_mat(j,2) - raw_data_mat(j - 1,2));
                end
            end
        end
        writing_time_index = j;
        rate_list(i - 1) = writing_time / (time_list(i) - time_list(i - 1));  
    end
    [max_value, max_index] = max(rate_list);
    rate_list = rate_list .* (100 / max(max_value, 0.0000000000000001));
    for i = 1:length(time_list) - 1
        fprintf(output_file, '%3.5f\r\n', rate_list(i));
    end
    slide_writing_mat = [slide_writing_mat; rate_list];
    fclose(output_file);
end

slide_writing_mat = slide_writing_mat';
for i = 1:length(slide_writing_mat(:,1))
    for j = 1:length(slide_writing_mat(1,:))
        fprintf(slide_writing_file, '%3.5f ', slide_writing_mat(i,j));
    end
    fprintf(slide_writing_file, '\r\n');
end
    