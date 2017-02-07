format long;
clear all;
Files = dir('wednesday/wifi-data/entropy_everyone_without_writing_normalized/');
% Files = dir('wednesday/wifi-data/entropy_everyone_without_writing/');
mat = {};
time_list = load('wednesday/time.txt');
slides_num = length(time_list) / 2;
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    raw_data_mat = load(strcat('wednesday/wifi-data/entropy_everyone_without_writing_normalized/',FileNames));
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
                [mat{j}{l}(:,1) mat{j}{l}(:,3)], time_list(2 * l - 1), time_list(2 * l), 30000);
            gyr_dist_matrix(i,j,l) = ...
                time_series_distance_calculator([mat{i}{l}(:,2) mat{i}{l}(:,3)],...
                [mat{j}{l}(:,2) mat{j}{l}(:,3)], time_list(2 * l - 1), time_list(2 * l), 30000);
        end
%         disp(strcat('j = ',int2str(j)));
    end
    disp(strcat('i = ',int2str(i)));
end

en_acc_dist_list = [];
en_gyr_dist_list = [];

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
    en_acc_dist_list = [en_acc_dist_list; acc_tmp];
    en_gyr_dist_list = [en_gyr_dist_list; gyr_tmp];
end

en_dist_mean_list = [mean(en_acc_dist_list,2), mean(en_gyr_dist_list,2)];
en_dist_std_list = [std(en_acc_dist_list')', std(en_gyr_dist_list')'];
en_dist_var_list = [var(en_acc_dist_list')', var(en_gyr_dist_list')'];
en_dist_rms_list = [rms(en_acc_dist_list,2), rms(en_gyr_dist_list,2)];
save('wednesday/wifi-data/entropy_distance_statistical_features_ww_n.mat',...
    'en_dist_mean_list', 'en_dist_std_list', 'en_dist_var_list', 'en_dist_rms_list');