Files = dir('monday/wifi-data/amplitude_without_writing/');
mat = [];
for k = 3:length(Files(:))
    FileNames = Files(k).name;
    raw_data_mat = load(strcat('monday/wifi-data/amplitude_without_writing/',FileNames));
    disp(FileNames);
    disp(length(raw_data_mat(:,1)));
    mat = [mat; raw_data_mat];
    
%     plot(acc_entropy,'y');
%     hold
%     plot(gyr_entropy, 'r');
end
disp(length(mat(:,1)));
% pause(3);

mat = sortrows(mat,3);

output_file = fopen('monday/wifi-data/entropy_all_without_writing.txt', 'a');
last_time = 0;
n = 0;
num = 0;
acc_list = [];
gyr_list = [];
for i = 1:length(mat(:,1))
    if mod(i,10000) == 0
        disp(i);
    end
    if last_time == 0
        last_time = mat(i,3);
        n = 1;
    else
        if mat(i,3) - last_time < 5000 && i < length(mat(:,1))
            n = n + 1;
        else
            num = num + 1;
            acc_entropy(num) = entropy_calculator(acc_list, 'acc');
            gyr_entropy(num) = entropy_calculator(gyr_list, 'gyr');
            fprintf(output_file, '%f %f %ld\r\n', acc_entropy(num), gyr_entropy(num), mat(i,3));
            n = 1;
            last_time = mat(i,3);
            acc_list = [];
            gyr_list = [];
        end
    end
    acc_list = [acc_list mat(i,1)];
    gyr_list = [gyr_list mat(i,2)];
    
end
fclose(output_file);
plot(acc_entropy,'y');
plot(gyr_entropy, 'r');