%   interval is 1000 ms
Files = dir('monday/wifi-data/amplitude_without_writing/');
mkdir('monday/wifi-data/motion_level_everyone_without_writing/');
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    output_file = fopen(strcat('monday/wifi-data/motion_level_everyone_without_writing/', FileNames), 'a');
    raw_data_mat = load(strcat('monday/wifi-data/amplitude_without_writing/',FileNames));
    last_time = 0;
    n = 0;
    num = 0;
    acc_list = [];
    gyr_list = [];
    for i = 1:length(raw_data_mat(:,1))
        if last_time == 0
            last_time = raw_data_mat(i,3);
            n = 1;
        else
            if raw_data_mat(i,3) - last_time < 1000 && i < length(raw_data_mat(:,1))
                n = n + 1;
            else
                num = num + 1;
                acc_motion_level(num) = motion_level_calculator(acc_list, 'acc');
                gyr_motion_level(num) = motion_level_calculator(gyr_list, 'gyr');
                fprintf(output_file, '%f %f %ld\r\n', acc_motion_level(num), gyr_motion_level(num), raw_data_mat(i,3));
                n = 1;
                last_time = raw_data_mat(i,3);
                acc_list = [];
                gyr_list = [];
            end
        end
        acc_list = [acc_list raw_data_mat(i,1)];
        gyr_list = [gyr_list raw_data_mat(i,2)];
                
    end
    fclose(output_file);
    
%     plot(acc_entropy,'y');
%     hold
%     plot(gyr_entropy, 'r');
end

function [ ret ] = motion_level_calculator(data_array, type)
acc_table_num = 40;
gyr_table_num = 60;

acc_table = zeros(1,acc_table_num);
gyr_table = zeros(1,gyr_table_num);
for i = 1:acc_table_num
    acc_table(i) = 0.1 + (i - 1) * 0.04;
end

for i = 1:gyr_table_num
    gyr_table(i) = 0.01 + (i - 1) * 0.07;
end

dis = [];
if type == 'acc'
    acc_mean = mean(data_array);
    %     disp(acc_mean);
    dis = acc_table - acc_mean;
    [minv,mini] = min(abs(dis));
    ret = acc_table(mini);
end
if type == 'gyr'
    gyr_mean = mean(data_array);
    dis = gyr_table - gyr_mean;
    [minv,mini] = min(abs(dis));
    ret = gyr_table(mini);
end
end