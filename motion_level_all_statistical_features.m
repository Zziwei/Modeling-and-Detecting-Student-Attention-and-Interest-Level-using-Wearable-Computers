clear all;
ml_all = load('monday/wifi-data/motion_level_all_without_writing.txt');
time_list = load('monday/time.txt');
slides_num = length(time_list) / 2;

ml_group_mean_list = zeros(slides_num,2);
ml_group_std_list = zeros(slides_num,2);
ml_group_rms_list = zeros(slides_num,2);
ml_group_var_list = zeros(slides_num,2);
ml_group_mcr_list = zeros(slides_num,2);
ml_group_skewness_list = zeros(slides_num,2);
ml_group_kurtosis_list = zeros(slides_num,2);

for i = 1:slides_num
    time_start = time_list(i * 2 - 1);
    time_end = time_list(i * 2);
    tmp_motionlevel_list = ml_all(find(time_start <= ml_all(:,3) & ml_all(:,3) <= time_end),:);
    ml_group_mean_list(i,1) = mean(tmp_motionlevel_list(:,1));
    ml_group_mean_list(i,2) = mean(tmp_motionlevel_list(:,2));
    ml_group_std_list(i,1) = std(tmp_motionlevel_list(:,1));
    ml_group_std_list(i,2) = std(tmp_motionlevel_list(:,2));
    ml_group_rms_list(i,1) = rms(tmp_motionlevel_list(:,1));
    ml_group_rms_list(i,2) = rms(tmp_motionlevel_list(:,2));
    ml_group_var_list(i,1) = var(tmp_motionlevel_list(:,1));
    ml_group_var_list(i,2) = var(tmp_motionlevel_list(:,2));
    ml_group_skewness_list(i,1) = skewness(tmp_motionlevel_list(:,1));
    ml_group_skewness_list(i,2) = skewness(tmp_motionlevel_list(:,2));
    ml_group_kurtosis_list(i,1) = kurtosis(tmp_motionlevel_list(:,1));
    ml_group_kurtosis_list(i,2) = kurtosis(tmp_motionlevel_list(:,2));
    for j = 1:(length(tmp_motionlevel_list(:,1)) - 1)
        if ((tmp_motionlevel_list(j,1) - ml_group_mean_list(i,1)) * (tmp_motionlevel_list(j + 1,1) - ml_group_mean_list(i,1))) < 0
            ml_group_mcr_list(i,1) = ml_group_mcr_list(i,1) + 1;
        end
        if ((tmp_motionlevel_list(j,2) - ml_group_mean_list(i,2)) * (tmp_motionlevel_list(j + 1,2) - ml_group_mean_list(i,2))) < 0
            ml_group_mcr_list(i,2) = ml_group_mcr_list(i,2) + 1;
        end
    end
    ml_group_mcr_list(i,1) = ml_group_mcr_list(i,1) / length(tmp_motionlevel_list(:,1) - 1);
    ml_group_mcr_list(i,2) = ml_group_mcr_list(i,2) / length(tmp_motionlevel_list(:,2) - 1);
end
