raw_data = load('wednesday/wifi-data/entropy_all.txt');
dateNumArray = (raw_data(:,3) ./86400000 + datenum(1970,1,1) - 6/24);
timeTextArray = datestr(raw_data(:,3) ./86400000 + datenum(1970,1,1) - 6/24);
timeArray = timeTextArray(:,13:20);
count = 1;
for i = 1:25:size(raw_data,1)
    timeReduced(count,:) = timeArray(i,:);
    count = count + 1;
end
figure;
time_list = load('wednesday/time.txt');
hold;
for t = 1:length(time_list(:,1))
    plot([(time_list(t,10) /86400000 + datenum(1970,1,1) - 6/24) (time_list(t,10) /86400000 + datenum(1970,1,1) - 6/24)], [1 4],'k');
end
plot(dateNumArray,raw_data(:,1));
set(gca,'XTick',dateNumArray(1:25:size(raw_data, 1)),'xticklabel',timeReduced);
title('Entropy of Acc for the group');
ylabel('Entropy');
% datetick('x',13,'keeplimits');

figure;
time_list = load('wednesday/time.txt');
hold;
for t = 1:length(time_list(:,1))
    plot([(time_list(t,10) /86400000 + datenum(1970,1,1) - 6/24) (time_list(t,10) /86400000 + datenum(1970,1,1) - 6/24)], [2 8],'k');
end
plot(dateNumArray,raw_data(:,2));
set(gca,'XTick',dateNumArray(1:25:size(raw_data, 1)),'xticklabel',timeReduced)
title('Entropy of gyroscope for the group');
ylabel('Entropy');
% datetick('x',13,'keeplimits');