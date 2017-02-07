format long g;
Files = dir('wednesday/wifi-data/data');
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    str_array = strsplit(FileNames, '.');
    id = str_array(1);
    raw = load(strcat('wednesday/wifi-data/data/',FileNames));
    
%     raw_data = raw(find((1480346103000.0 <= raw(:,10)) & (raw(:,10) <= 1480348633000.0)),:);
    raw_data = raw(find((1480519021278.0 <= raw(:,10)) & (raw(:,10) <= 1480520968677.0)),:);
    
    acc_amp = (raw_data(:,1) .^ 2 + raw_data(:,2) .^ 2 + raw_data(:,3) .^ 2) .^ 0.5;
    gyr_amp = (raw_data(:,4) .^ 2 + raw_data(:,5) .^ 2 + raw_data(:,6) .^ 2) .^ 0.5;
    
    train_set = {};
    n = 0;
    for i = 1:length(raw_data(:,1)) - 74
        n = n + 1;
        train_set{n} =  [raw_data(i:(i+74),1:6) acc_amp(i:(i+74)) gyr_amp(i:(i+74)) raw_data(i:(i+74),10)];
    end
    
    tmp = train_set{1};
    
    mean_list = zeros(n,8);
%     std_list = zeros(n,8);
%     rms_list = zeros(n,8);
%     var_list = zeros(n,8);
    mcr_list = zeros(n,8);
%     skewness_list = zeros(n,8);
%     kurtosis_list = zeros(n,8);
%     zcr_list = zeros(n,6);
    
    for i = 1:n
        current_data = train_set{i};
        for j = 1:8
            mean_list(i,j) = mean(current_data(:,j));
%             std_list(i,j) = std(current_data(:,j));
%             rms_list(i,j) = rms(current_data(:,j));
%             var_list(i,j) = var(current_data(:,j));
%             skewness_list(i,j) = skewness(current_data(:,j));
%             kurtosis_list(i,j) = kurtosis(current_data(:,j));
            for l = 1:(length(current_data(:,j)) - 1)
                if ((current_data(l,j) - mean_list(i,j)) * (current_data(l + 1,j) - mean_list(i,j))) < 0
                    mcr_list(i,j) = mcr_list(i,j) + 1;
                end
            end
            mcr_list(i,j) = mcr_list(i,j) / (length(current_data(:,j)) - 1);
        end
    end
    
    features_mat = [];
    for i = 1:n
%         features_mat(i,:) = [mean_list(i,:), std_list(i,:), rms_list(i,:), var_list(i,:), skewness_list(i,:), kurtosis_list(i,:), mcr_list(i,:)];
        features_mat(i,:) = [mean_list(i,:), mcr_list(i,:), train_set{i}(1,9), train_set{i}(end,9)];
    end
    csv_file_name = (strcat(strcat('wednesday/wifi-data/writing_features_everyone/', id), '.csv'));
%     csvwrite(csv_file_name{1}, features_mat);
    dlmwrite(csv_file_name{1}, features_mat, 'delimiter', ',', 'precision', '%i'); 
    disp(id);
end