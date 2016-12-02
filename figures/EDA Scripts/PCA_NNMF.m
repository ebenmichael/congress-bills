%% EDA on votes data
clear
figure(1); clf

% load data
votes = load('votes.dat'); 
senator_MD = loadjson('senator_metadata.json');
senator_MD = struct2cell(senator_MD); 
membership = load('membership.dat'); 

% remove dead ppl (senators w multiple -1s) 
for i = 1 : size(votes, 2)
     if sum(votes(:,i)==-1) == 0; 
         keep(i) = logical(1); 
     end
end

votes_Trim = votes(:,keep); 

% set absent, abstaining, etc to 0
votes_Trim(votes_Trim~=0 & votes_Trim~=1) = 0 ; 

Party   = zeros(length(senator_MD), 1); 
Caucus1 = zeros(length(senator_MD), 1); 


% set up party labels for plots
for i = 1:length(senator_MD)
    if senator_MD{i}.party == 'D'
        Party(i) = 'D'; 
    end
    
    if senator_MD{i}.party == 'R'
        Party(i) = 'R'; 
    end 
    
    % caucuses (from liberal to conservative): 
    % Congressional Progressive Caucus: 270
    % New Democrat Coalition: 145
    % Blue Dog Coalition: 31
    % Tuesday Group: 200
    % Main street partnership: 583
    % Republical Study Committee: 181

    
    if membership(i,31) == 1
        Caucus1(i) = 1; 
    elseif membership(i,145) == 1
        Caucus1(i) = 2;
    elseif membership(i, 270) == 1 
        Caucus1(i) = 3;
    elseif membership(i,200) == 1
        Caucus1(i) = 4; 
    elseif membership(i,339) == 1
        Caucus1(i) = 5; 
    elseif membership(i, 181) == 1
        Caucus1(i) = 6;
    else 
        Caucus1(i) = 7; 
    end 
        
end

Party = Party(keep); 
Caucus1 = Caucus1(keep); 

%% nonnegative matrix factorization
figure(1); clf
[bills, senators] = nnmf(votes_Trim, 2);

scatter(senators(1,Party=='D'), senators(2,Party=='D'), 'b'); 
hold on 
scatter(senators(1,Party=='R'), senators(2,Party=='R'), 'r'); 

title('NNMF on roll call votes')
xlabel('x1')
ylabel('x2')
hold off 
legend('Dem', 'Rep')


figure(3); clf
scatter(senators(1,Caucus1 == 1),senators(2,Caucus1 == 1), 'yo', 'filled')
hold on
scatter(senators(1,Caucus1 == 2),senators(2,Caucus1 == 2), 'bo', 'filled')
scatter(senators(1,Caucus1 == 3),senators(2,Caucus1 == 3), 'co', 'filled')
scatter(senators(1,Caucus1 == 4),senators(2,Caucus1 == 4), 'go', 'filled')
scatter(senators(1,Caucus1 == 5),senators(2,Caucus1 == 5), 'mo', 'filled')
scatter(senators(1,Caucus1 == 6),senators(2,Caucus1 == 6), 'ro', 'filled')

%scatter(senators(1,Caucus1 == 7),senators(2,Caucus1 == 7), 'k')
legend('Blue Dog Coalition', 'New Democratic Caucus', ...
    'Congressional Progressive Caucus', 'Tuesday Group', ...
    'Main Street Partnership', 'Republican Study Committee')
title('NNMF on Caucus Membership')
xlabel('x1')
ylabel('x2')


%% PCA 
figure(2); clf
Sigma = cov(votes_Trim'); 

[U E V] = svd(Sigma); 

pca = U(:,1:2)' * votes_Trim; 

scatter(pca(1,Party=='D'), pca(2,Party=='D'), 'b'); 
hold on 
scatter(pca(1,Party=='R'), pca(2,Party=='R'), 'r'); 

title('PCA on roll call votes')
xlabel('PCA1')
ylabel('PCA2')
hold off 

legend('Dem', 'Rep')

figure(4); clf
scatter(pca(1,Caucus1 == 1),pca(2,Caucus1 == 1), 'yo', 'filled')
hold on
scatter(pca(1,Caucus1 == 2),pca(2,Caucus1 == 2), 'bo', 'filled')
scatter(pca(1,Caucus1 == 3),pca(2,Caucus1 == 3), 'ro', 'filled')
scatter(pca(1,Caucus1 == 4),pca(2,Caucus1 == 4), 'go', 'filled')
scatter(pca(1,Caucus1 == 5),pca(2,Caucus1 == 5), 'mo', 'filled')
scatter(pca(1,Caucus1 == 6),pca(2,Caucus1 == 6), 'co', 'filled')

%scatter(pca(1,Caucus1 == 7),pca(2,Caucus1 == 7), 'w')

legend('Blue Dog Coalition', 'New Democratic Caucus', ...
    'Congressional Progressive Caucus', 'Tuesday Group', ...
    'Main Street Partnership', 'Republican Study Committee')

title('PCA on Caucus Membership')
xlabel('PCA1')
ylabel('PCA2')
hold off 



