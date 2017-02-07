Files=dir('monday/wifi-data/data_without_writing/');
mkdir('monday/wifi-data/amplitude_without_writing/');
for k=3:length(Files(:))
    FileNames=Files(k).name;
    output_file = fopen(strcat('monday/wifi-data/amplitude_without_writing/',FileNames), 'a');
    raw_data_mat = load(strcat('monday/wifi-data/data_without_writing/',FileNames));
    amplitude_acc = (raw_data_mat(:,1).^2 + raw_data_mat(:,2).^2 + raw_data_mat(:,3).^2).^0.5;
    amplitude_gyr = (raw_data_mat(:,4).^2 + raw_data_mat(:,5).^2 + raw_data_mat(:,6).^2).^0.5;
    for i = 1:length(amplitude_acc(:,1))
        if (1480346103000.0 <= raw_data_mat(i,7) && raw_data_mat(i,7) <= 1480348633000.0)
%         if (1480519021278.0 <= raw_data_mat(i,7) && raw_data_mat(i,7) <= 1480520968677.0)
            fprintf(output_file, '%f %f %ld \r\n', amplitude_acc(i,1), amplitude_gyr(i,1), raw_data_mat(i,7));
        end
    end
    fclose(output_file);
end