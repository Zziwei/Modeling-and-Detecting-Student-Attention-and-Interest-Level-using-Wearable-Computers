load('entropyWorkspace.mat');

timeTextArray = datestr(timeStamps./86400000 + datenum(1970,1,1) - 7/24);

timeArray = timeTextArray(:,13:20);

count = 1;
for i = 1:50:size(entropy,1)
    timeReduced(count,:) = timeArray(i,:);
    count = count+1;
end

figure;
plot(entropy);
set(gca,'XTick',1:50:size(entropy,1),'xticklabel',timeReduced)
title('Collective Watch Entropy over Time');
ylabel('Entropy (bits)');