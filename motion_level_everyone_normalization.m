clear;
time_list = load('wednesday/non_duplicate_time.txt');
Files = dir('wednesday/wifi-data/motion_level_everyone_without_writing/');
mkdir('wednesday/wifi-data/motion_level_everyone_without_writing_normalized/');
mixed_slide_set = {};
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    output_file = fopen(strcat('wednesday/wifi-data/motion_level_everyone_without_writing_normalized/', FileNames), 'a');
    raw_data_mat = load(strcat('wednesday/wifi-data/motion_level_everyone_without_writing/',FileNames));
    max_value_acc = max(raw_data_mat(:,1));
    max_value_gyr = max(raw_data_mat(:,2));
    raw_data_mat(:,1) = raw_data_mat(:,1) .* (2 / max_value_acc);
    raw_data_mat(:,2) = raw_data_mat(:,2) .* (5 / max_value_gyr);
    for i = 1:length(raw_data_mat(:,1))
        fprintf(output_file, '%f %f %ld\r\n', raw_data_mat(i,1), raw_data_mat(i,2), raw_data_mat(i,3));
    end
    fclose(output_file);
    for i = 2:length(time_list)
        if k == 3
            mixed_slide_set{i - 1} = [];
        end
        mixed_slide_set{i - 1} = [mixed_slide_set{i - 1},...
            raw_data_mat(time_list(i - 1) <= raw_data_mat(:,3) & time_list(i) >= raw_data_mat(:,3),1:2)'];
    end
    
%     plot(acc_entropy,'y');
%     hold
%     plot(gyr_entropy, 'r');
end

ml_everyone_normalized_mean_list = zeros(length(time_list) - 1,2);
ml_everyone_normalized_std_list = zeros(length(time_list) - 1,2);
ml_everyone_normalized_var_list = zeros(length(time_list) - 1,2);
ml_everyone_normalized_rms_list = zeros(length(time_list) - 1,2);

for i = 1:length(time_list) - 1
    ml_everyone_normalized_mean_list(i,1) = mean(mixed_slide_set{i}(1,:));
    ml_everyone_normalized_std_list(i,1) = std(mixed_slide_set{i}(1,:));
    ml_everyone_normalized_var_list(i,1) = var(mixed_slide_set{i}(1,:));
    ml_everyone_normalized_rms_list(i,1) = rms(mixed_slide_set{i}(1,:));
    ml_everyone_normalized_mean_list(i,2) = mean(mixed_slide_set{i}(2,:));
    ml_everyone_normalized_std_list(i,2) = std(mixed_slide_set{i}(2,:));
    ml_everyone_normalized_var_list(i,2) = var(mixed_slide_set{i}(2,:));
    ml_everyone_normalized_rms_list(i,2) = rms(mixed_slide_set{i}(2,:));
end

save('wednesday/wifi-data/motion_level_everyone_normalized_statistical_features.mat',...
    'ml_everyone_normalized_mean_list', 'ml_everyone_normalized_std_list',...
    'ml_everyone_normalized_var_list', 'ml_everyone_normalized_rms_list');
