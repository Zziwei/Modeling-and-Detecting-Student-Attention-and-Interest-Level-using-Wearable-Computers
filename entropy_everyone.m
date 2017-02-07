%   interval is 5000ms
Files = dir('monday/wifi-data/amplitude_without_writing/');
mkdir('monday/wifi-data/entropy_everyone_without_writing/');
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    output_file = fopen(strcat('monday/wifi-data/entropy_everyone_without_writing/', FileNames), 'a');
    raw_data_mat = load(strcat('monday/wifi-data/amplitude_without_writing/',FileNames));
    last_time = 0;
    n = 0;
    num = 0;
    acc_list = [];
    gyr_list = [];
    for i = 1:length(raw_data_mat(:,1))
%         if raw_data_mat(i,3) < 1480346103000
%             continue;
%         end
%         
%         if raw_data_mat(i,3) > 1480348633000
%             break;
%         end
        
        if last_time == 0
            last_time = raw_data_mat(i,3);
            n = 1;
        else
            if raw_data_mat(i,3) - last_time < 5000 && i < length(raw_data_mat(:,1))
                n = n + 1;
            else
                num = num + 1;
                acc_entropy(num) = entropy_calculator(acc_list, 'acc');
                gyr_entropy(num) = entropy_calculator(gyr_list, 'gyr');
                fprintf(output_file, '%f %f %ld\r\n', acc_entropy(num), gyr_entropy(num), raw_data_mat(i,3));
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