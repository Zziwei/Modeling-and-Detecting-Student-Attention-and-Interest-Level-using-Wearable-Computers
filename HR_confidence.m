Files = dir('Data by Slide\');
HR_features_list = zeros(14,4);
HRV_features_list = zeros(14,32);

for k = 3:length(Files(:))
    HRV_set = {}
    for i = 1:8
        HRV_set{i} = [];
    end
    FileNames = Files(k).name;
    raw_data_mat = csvread(strcat('Data by Slide/',FileNames));
    disp(FileNames);
    confidence = raw_data_mat(2,:) .* (1 / max(raw_data_mat(2,:)));
    HR_features_list(k - 2,1) = mean(raw_data_mat(1,:) .* confidence);
    HR_features_list(k - 2,2) = std(raw_data_mat(1,confidence >= 0.6));
    HR_features_list(k - 2,3) = var(raw_data_mat(1,confidence >= 0.6));
    HR_features_list(k - 2,4) = rms(raw_data_mat(1,confidence >= 0.6));
    for i = 1:((length(raw_data_mat(:,1) - 2) / 10))
        j = 2 + 3 + (i - 1) * 10;
        for l = 1:8
            HRV_set{l} = [HRV_set{l}, raw_data_mat(j + l - 1, (raw_data_mat(j + l - 1, :) > 0))];
        end
    end
    for i = 1:8
        HRV_features_list(k - 2,1 + (i - 1) * 4) = mean(HRV_set{i});
        HRV_features_list(k - 2,2 + (i - 1) * 4) = std(HRV_set{i});
        HRV_features_list(k - 2,3 + (i - 1) * 4) = var(HRV_set{i});
        HRV_features_list(k - 2,4 + (i - 1) * 4) = rms(HRV_set{i});
    end
end

input_d = csvread('input_d_2.csv');
input_i = csvread('input_i_2.csv');
input_d = [HRV_features_list HR_features_list input_d];
input_i = [ HRV_features_list HR_features_list input_i];
csvwrite('input_d_2_hrv_c.csv',input_d);
csvwrite('input_i_2_hrv_c.csv',input_i);