%% EDA on votes data
% Plot membership against votes

clear
figure(1); clf

%% load data
votes = load('votes.dat'); % row are n bills; columns are s senators

% remove dead ppl (senators w multiple -1s) 
for i = 1 : size(votes, 2)
     if sum(votes(:,i)==-1) == 0; 
         keep(i) = logical(1); 
     end
end

votes = votes(:,keep); 
votes(votes~=0 & votes~=1) = NaN; 

% load senetors
senator_MD = loadjson('senator_metadata.json');
senator_MD = struct2cell(senator_MD); 

% load membership
Membership = load('membership.dat'); 
Membership = Membership(keep,:); 

Membership(isnan(Membership)) = 0; % set NaNs to 0

% count the number of times i is in same caucus as j
% count the number of times i voted the same as j

CoMship = zeros(1,size(Membership,1)*size(Membership,1)-1); 
CoBill = zeros(1,size(Membership,1)*size(Membership,1)-1); 
k=0; 
for i = 1:size(Membership,1)
    for j = 1:i-1
        k=k+1; 
        CoMship(k) = Membership(i,:) * Membership(j,:)'; 
        CoBill(k) = sum(votes(:,i) == votes(:,j))...
            /max(sum(votes(:,i)==votes(:,i)), sum(votes(:,j)==votes(:,j))); 
end
end

% Party = cell(length(senator_MD), 1); 
% 
% % set up party labels for plots
% for i = 1:length(senator_MD)
%     if senator_MD{i}.party == 'D'
%         Party{i} = 'b'; 
%     end
%     
%     if senator_MD{i}.party == 'R'
%         Party{i} = 'r'; 
%     end 
%     
%     % caucuses (from liberal to conservative): 
%     % Congressional Progressive Caucus: 270
%     % New Democrat Coalition: 145
%     % Blue Dog Coalition: 31
%     % Tuesday Group: 200
%     % Main street partnership: 583
%     % Republical Study Committee: 181
% 
%     
% %     if Mship(i,31) == 1
% %         Caucus1{i} = 'Blue Dog Coalition'; 
% %     elseif Mship(i,145) == 1
% %         Caucus1{i} = 'New Democrat Coalition';
% %     elseif Mship(i, 270) == 1 
% %         Caucus1{i} = 'Congressional Progressive Caucus';
% %     elseif Mship(i,200) == 1
% %         Caucus1{i} = 'Tuesday Group'; 
% %     elseif Mship(i,339) ==1
% %         Caucus1{i} = 'Main street partnership'; 
% %     elseif Mship(i, 181) == 1
% %         Caucus1{i} = 'Repubulican Study Committee';
% %     else 
% %         Caucus1{i} = 'Neither'; 
% %     end 
%         
% end
% 
% Label = Party;     

scatter(CoMship, CoBill, '.')
xlabel('Number of shared caucuses')
ylabel('Proportion of agreement on bills')

figure(2); clf

% boxplot at 30 only has two data points; I don't want this point
% to be a boxplot; convert them to a scatter plot
edits1 = CoBill(CoMship~=30); 
edits2 = CoMship(CoMship~=30); 
edits1(end+1) = 0.9257; 
edits2(end+1) = 30; 

boxplot(edits1, edits2)
hold on 
scatter(31, 0.5211, 'r+')
xlabel('Number of shared caucuses')
ylabel('Proportion of agreement on bills')
set(gca,'XLim',[0 35])
set(gca,'YLim',[0,1])
