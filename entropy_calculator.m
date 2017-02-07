function [ ret ] = entropy_calculator(data_array, type)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
acc_table_num = 30;
gyr_table_num = 300;

acc_table = zeros(1,acc_table_num);
gyr_table = zeros(1,gyr_table_num);
for i = 1:acc_table_num
    acc_table(i) = 0.1 + (i - 1) * 0.04;
end

for i = 1:gyr_table_num
    gyr_table(i) = 0.01 + (i - 1) * 0.013;
end

acc_dis = zeros(1,acc_table_num + 1);
gyr_dis = zeros(1,gyr_table_num + 1);

if type == 'acc'
    for i = 1:length(data_array)
        is_in = false;
        for j = 1:length(acc_table)
            if data_array(i) <= acc_table(j)
                acc_dis(j) = acc_dis(j) + 1;
                is_in = true;
                break;
            end
        end
        if is_in == false
            acc_dis(acc_table_num + 1) = acc_dis(acc_table_num + 1) + 1;
        end
    end
    ret = 0;
    for i = 1:length(acc_dis)
        p = acc_dis(i) / length(data_array);
        if p == 0
            continue;
        end
        ret = ret - (p * log2(p));
    end
end

if type == 'gyr'
    for i = 1:length(data_array)
        is_in = false;
        for j = 1:length(gyr_table)
            if data_array(i) <= gyr_table(j)
                gyr_dis(j) = gyr_dis(j) + 1;
                is_in = true;
                break;
            end
        end
        if is_in == false
            gyr_dis(length(gyr_dis)) = gyr_dis(length(gyr_dis)) + 1;
        end
    end
    ret = 0;
    for i = 1:length(gyr_dis)
        p = gyr_dis(i) / length(data_array);
        if p == 0
            continue;
        end
        ret = ret - (p * log2(p));
    end
end
end

