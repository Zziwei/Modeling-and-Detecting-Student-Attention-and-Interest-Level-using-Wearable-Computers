survey_raw = load('monday/survey1.txt');
survey = zeros(16,2);
for i = 1:30
   for j = 1:8
        survey(j,1) = survey(j,1) + survey_raw((j + 1) / 2 + 8 * (i - 1),1);
        survey(j,2) = survey(j,2) + survey_raw((j + 1) / 2 + 8 * (i - 1),2);
        
   end
end

hold;
plotyy(entropy_all(:,3),entropy_all(:,1),[time_list(:,10) time_list(:,10)],survey);
for t = 1:length(time_list(:,1))
    plot([time_list(t,10) time_list(t,10)], [1.25 2.5]);
end
ylim([1.25 2.5]);