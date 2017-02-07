load('writing_recognition_dtw_data.mat');

test_dis = zeros(1,100);
for i = 1:100
    for j = 1:640
        test_dis(i) = test_dis(i) + dtw(test_set{i}',train_set{j}');
    end
end
test_dis = test_dis ./ 640;
error_test = length(find(test_dis > 66));

outlier_dis = zeros(1,100);
for i = 1:100
    for j = 1:640
        outlier_dis(i) = outlier_dis(i) + dtw(outlier_set{i}',train_set{j}');
    end
end
outlier_dis = outlier_dis ./ 640;
error_outlier = length(find(outlier_dis < 66));