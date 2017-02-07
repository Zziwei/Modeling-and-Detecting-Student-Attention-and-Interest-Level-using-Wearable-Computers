Files = dir('wednesday/wifi-data/writing_results_everyone/');
for k = 3:length(Files(:))
    close all;
    FileNames = Files(k).name;
%     FileNames = '2L2S.txt';
    raw_data = load(strcat('wednesday/wifi-data/writing_results_everyone/',FileNames));
    merged_result = [];
    if raw_data(1,1) < 0
        is_last_positive = false;
    else
        is_last_positive = true;
    end
    start_t = raw_data(1,2);
    end_t = raw_data(1,3);
    for i = 2:length(raw_data(:,1))
        if raw_data(i,1) < 0
            if is_last_positive == true
                if end_t - start_t >= 1000
                    merged_result = [merged_result; 1, start_t, end_t];
                end
                start_t = end_t;
                end_t = raw_data(i,3);
                is_last_positive = false;
            else
                end_t = raw_data(i,3);
            end
        else
            if is_last_positive == false
                if raw_data(i,2) - start_t >= 3000
                    merged_result = [merged_result; -1, start_t, raw_data(i,2)];
                end
                start_t = raw_data(i,2);
                end_t = raw_data(i,3);
                is_last_positive = true;
            else
                end_t = raw_data(i,3);
            end
        end
    end
    if is_last_positive == true
        merged_result = [merged_result; 1, start_t, end_t];
    else
        merged_result = [merged_result; -1, start_t, end_t];
    end
    
    output_mat = [];
    if merged_result(1,1) < 0
        is_last_positive = false;
    else
        is_last_positive = true;
    end
    start_t = merged_result(1,2);
    end_t = merged_result(1,3);
    for i = 2:length(merged_result(:,1))
        if merged_result(i,1) < 0
            if is_last_positive == true
                output_mat = [output_mat; 1, start_t, end_t];
                start_t = merged_result(i,2);
                end_t = merged_result(i,3);
                is_last_positive = false;
            else
                end_t = merged_result(i,3);
            end
        else
            if is_last_positive == false
                output_mat = [output_mat; -1, start_t, end_t];
                start_t = merged_result(i,2);
                end_t = merged_result(i,3);
                is_last_positive = true;
            else
                end_t = merged_result(i,3);
            end
        end
    end
    if is_last_positive == true
        output_mat = [output_mat; 1, start_t, end_t];
    else
        output_mat = [output_mat; -1, start_t, end_t];
    end
    
    output_file = fopen(strcat('wednesday/wifi-data/writing_merged_results_everyone/',FileNames), 'a');
    n = 0;
    for j = 1:length(output_mat(:,1))
        fprintf(output_file, '%d %ld \r\n', output_mat(j,1), output_mat(j,2));
        fprintf(output_file, '%d %ld \r\n', output_mat(j,1), output_mat(j,3));
        n = n + 1;
    end
    fclose(output_file);  
    disp(n);
end