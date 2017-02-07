function [ dist ] = time_series_distance_calculator( series1, series2, start_t, end_t, window_size )
%TIME_SERIES_DISTANCE_CALCULATOR Summary of this function goes here
%   Detailed explanation goes here
    format long;
    dist = 0;
    skip = 0;
    for i = start_t:window_size:end_t
        current_sub_series_1 = series1(series1(:,2) >= i & series1(:,2) < (i + window_size - 1), 1);
        current_sub_series_2 = series2(series2(:,2) >= i & series2(:,2) < (i + window_size - 1), 1);
        if length(current_sub_series_1) == 0 || length(current_sub_series_2) == 0
            skip = skip + 1;
            continue;
        end
        dist = dist + (mean(current_sub_series_1) - mean(current_sub_series_2)) ^ 2;
    end
    if i < end_t
        current_sub_series_1 = series1(find(series1(:,2) >= i & series1(:,2) < (i + window_size - 1)), 1);
        current_sub_series_2 = series2(find(series2(:,2) >= i & series2(:,2) < (i + window_size - 1)), 1);
        if length(current_sub_series_1) ~= 0 && length(current_sub_series_2) ~= 0
            dist = dist + (mean(current_sub_series_1) - mean(current_sub_series_2)) ^ 2;
        end
    end
    dist = dist ^ 0.5;
    disp(skip);
end

