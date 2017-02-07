clear all;
entropy_all = load('monday/wifi-data/entropy_all_without_writing.txt');
time_list = load('monday/time.txt');
slides_num = length(time_list) / 2;

en_group_mean_list = zeros(slides_num,2);
en_group_std_list = zeros(slides_num,2);
en_group_rms_list = zeros(slides_num,2);
en_group_var_list = zeros(slides_num,2);
en_group_mcr_list = zeros(slides_num,2);
en_group_skewness_list = zeros(slides_num,2);
en_group_kurtosis_list = zeros(slides_num,2);

for i = 1:slides_num
    time_start = time_list(i * 2 - 1);
    time_end = time_list(i * 2);
    tmp_entropy_list = entropy_all(find(time_start<= entropy_all(:,3) & entropy_all(:,3) <= time_end),:);
    en_group_mean_list(i,1) = mean(tmp_entropy_list(:,1));
    en_group_mean_list(i,2) = mean(tmp_entropy_list(:,2));
    en_group_std_list(i,1) = std(tmp_entropy_list(:,1));
    en_group_std_list(i,2) = std(tmp_entropy_list(:,2));
    en_group_rms_list(i,1) = rms(tmp_entropy_list(:,1));
    en_group_rms_list(i,2) = rms(tmp_entropy_list(:,2));
    en_group_var_list(i,1) = var(tmp_entropy_list(:,1));
    en_group_var_list(i,2) = var(tmp_entropy_list(:,2));
    en_group_skewness_list(i,1) = skewness(tmp_entropy_list(:,1));
    en_group_skewness_list(i,2) = skewness(tmp_entropy_list(:,2));
    en_group_kurtosis_list(i,1) = kurtosis(tmp_entropy_list(:,1));
    en_group_kurtosis_list(i,2) = kurtosis(tmp_entropy_list(:,2));
    for j = 1:(length(tmp_entropy_list(:,1)) - 1)
        if ((tmp_entropy_list(j,1) - en_group_mean_list(i,1)) * (tmp_entropy_list(j + 1,1) - en_group_mean_list(i,1))) < 0
            en_group_mcr_list(i,1) = en_group_mcr_list(i,1) + 1;
        end
        if ((tmp_entropy_list(j,2) - en_group_mean_list(i,2)) * (tmp_entropy_list(j + 1,2) - en_group_mean_list(i,2))) < 0
            en_group_mcr_list(i,2) = en_group_mcr_list(i,2) + 1;
        end
    end
    en_group_mcr_list(i,1) = en_group_mcr_list(i,1) / length(tmp_entropy_list(:,1) - 1);
    en_group_mcr_list(i,2) = en_group_mcr_list(i,2) / length(tmp_entropy_list(:,2) - 1);
end
