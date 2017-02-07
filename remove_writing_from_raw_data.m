Files = dir('monday/wifi-data/data');
mkdir('monday/wifi-data/data_without_writing/');
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    disp(FileNames);
    raw_data_mat = load(strcat('monday/wifi-data/data/', FileNames));
    writing_mat = load(strcat('monday/wifi-data/writing_merged_results_everyone/', FileNames));
    writing_mat = writing_mat(writing_mat(:,1) == 1, :);
    if isempty(writing_mat)
        w_start_t = raw_data_mat(end,10) + 1;
    else
        w = 1;
        w_start_t = writing_mat(w,2);
        w_end_t = writing_mat(w + 1,2);
    end
    output_file = fopen(strcat('monday/wifi-data/data_without_writing/', FileNames), 'a');
    i = 1;
    while i <= length(raw_data_mat(:,1))
        if raw_data_mat(i,10) >= w_start_t
            for j = i:length(raw_data_mat(:,1))
                if raw_data_mat(j,10) > w_end_t
                    break;
                end
            end
            i = j - 1;
            w = w + 2;
            if w <= length(writing_mat(:,1))
                w_start_t = writing_mat(w,2);
                w_end_t = writing_mat(w + 1,2);
            else
                w_start_t = raw_data_mat(end,10) + 1;
            end
            continue;
        end
        fprintf(output_file, '%f %f %f %f %f %f %ld \r\n', raw_data_mat(i,1),...
            raw_data_mat(i,2), raw_data_mat(i,3), raw_data_mat(i,4),...
            raw_data_mat(i,5), raw_data_mat(i,6), raw_data_mat(i,10));
        i = i + 1;
    end
    fclose(output_file);
end