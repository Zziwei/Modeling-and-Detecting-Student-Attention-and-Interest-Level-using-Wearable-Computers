close all;
clear;
format long;
Files = dir('monday/wifi-data/motion_level_everyone_without_writing_normalized/');
% Files = dir('wednesday/wifi-data/motion_level_everyone_without_writing/');
mat = {};
time_list = load('monday/time.txt');
slides_num = length(time_list) / 2;
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    raw_data_mat = load(strcat('monday/wifi-data/motion_level_everyone_without_writing_normalized/',FileNames));
    disp(FileNames);
    classes = {};
    for t = 1:2:length(time_list)
        classes{(t + 1) / 2} = raw_data_mat(raw_data_mat(:,3) >= time_list(t) & raw_data_mat(:,3) <= time_list(t + 1),:);
    end
    mat{k - 2} = classes;
end

acc_dist_matrix = zeros(length(mat),length(mat),slides_num);
gyr_dist_matrix = zeros(length(mat),length(mat),slides_num);
for i = 1:length(mat)
    for j = 1:length(mat)
        for l = 1:slides_num
            acc_dist_matrix(i,j,l) = ...
                time_series_distance_calculator([mat{i}{l}(:,1) mat{i}{l}(:,3)],...
                [mat{j}{l}(:,1) mat{j}{l}(:,3)], time_list(2 * l - 1), time_list(2 * l), 20000);
            gyr_dist_matrix(i,j,l) = ...
                time_series_distance_calculator([mat{i}{l}(:,2) mat{i}{l}(:,3)],...
                [mat{j}{l}(:,2) mat{j}{l}(:,3)], time_list(2 * l - 1), time_list(2 * l), 20000);
        end
%         disp(strcat('j = ',int2str(j)));
    end
    disp(strcat('i = ',int2str(i)));
end

ml_acc_dist_list = [];
ml_gyr_dist_list = [];

for i = 1:slides_num
    disp(i);
    acc_tmp = [];
    gyr_tmp = [];
    for j = 1:length(mat)
        for l = 1:length(mat)
            if j == l
                continue;
            end
            acc_tmp = [acc_tmp acc_dist_matrix(j,l,i)];
            gyr_tmp = [gyr_tmp gyr_dist_matrix(j,l,i)];
        end
    end
    ml_acc_dist_list = [ml_acc_dist_list; acc_tmp];
    ml_gyr_dist_list = [ml_gyr_dist_list; gyr_tmp];
end

ml_dist_mean_list = [mean(ml_acc_dist_list,2), mean(ml_gyr_dist_list,2)];
ml_dist_std_list = [std(ml_acc_dist_list')', std(ml_gyr_dist_list')'];
ml_dist_var_list = [var(ml_acc_dist_list')', var(ml_gyr_dist_list')'];
ml_dist_rms_list = [rms(ml_acc_dist_list,2), rms(ml_gyr_dist_list,2)];
save('monday/wifi-data/motion_level_distance_statistical_features_ww_n.mat',...
    'ml_dist_mean_list', 'ml_dist_std_list', 'ml_dist_var_list', 'ml_dist_rms_list');