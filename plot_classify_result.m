close all;
result = load('monday\classify_result_3classes.txt');
bar(result .* 100);
Labels = {'decision tree', 'knn', 'svm', 'naive bayes'};
set(gca, 'XTick', 1:4, 'XTickLabel', Labels);
legend('mean','std');
ylabel('Accuracy');
xlabel('Classifiers')

labels=get(gca,'YTickLabel'); % get the y axis labels

for i=1:size(labels,1)
   labels_modif(i,:)=strcat(labels(i,:), ' %');
end
set(gca,'YTickLabel',labels_modif);