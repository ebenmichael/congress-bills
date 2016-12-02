%% EDA on votes data
% Create matrix with caucus co-membership, perform PCA

clear
figure(1); clf

% load data
senator_MD = loadjson('senator_metadata.json');
senator_MD = struct2cell(senator_MD); 
Mship = load('membership.dat'); 

Mship(isnan(Mship)) = 0; % set NaNs to 0

CoMship = ones(size(Mship,1),size(Mship,1)); 

% count the number of times i is in same caucus as j
for i = 1:size(Mship,1)
    for j = 1:size(Mship,1) 
    CoMship(i,j) = Mship(i,:) * Mship(j,:)'; 
end
end

Party = cell(length(senator_MD), 1); 

% set up party labels for plots
for i = 1:length(senator_MD)
    if senator_MD{i}.party == 'D'
        Party{i} = 'b'; 
    end
    
    if senator_MD{i}.party == 'R'
        Party{i} = 'r'; 
    end 
    
    % caucuses (from liberal to conservative): 
    % Congressional Progressive Caucus: 270
    % New Democrat Coalition: 145
    % Blue Dog Coalition: 31
    % Tuesday Group: 200
    % Main street partnership: 583
    % Republical Study Committee: 181

    
%     if Mship(i,31) == 1
%         Caucus1{i} = 'Blue Dog Coalition'; 
%     elseif Mship(i,145) == 1
%         Caucus1{i} = 'New Democrat Coalition';
%     elseif Mship(i, 270) == 1 
%         Caucus1{i} = 'Congressional Progressive Caucus';
%     elseif Mship(i,200) == 1
%         Caucus1{i} = 'Tuesday Group'; 
%     elseif Mship(i,339) ==1
%         Caucus1{i} = 'Main street partnership'; 
%     elseif Mship(i, 181) == 1
%         Caucus1{i} = 'Repubulican Study Committee';
%     else 
%         Caucus1{i} = 'Neither'; 
%     end 
        
end

Label = Party;     

%% nonnegative matrix factorization
% [bills, senators] = nnmf(votes_Trim, 2);
% 
% gscatter(senators(1,:), senators(2,:), Label, 'bkrbbrr', '+.+dood'); 
% 
% % scatter(senators, 1, [], color); 
% 
% %hold on
% %scatter(bills(:,1), bills(:,2), 'r')
% %hold off
% title('Nonnegative matrix factorization')
% xlabel('x1')
% ylabel('x2')



%% PCA 
figure(2); clf

[U E V] = svd(CoMship); 

pca = U(:,1:2); 

scatter(pca(:,1)', pca(:,2)', [], Label); 

title('PCA')
xlabel('PCA1')
ylabel('PCA2')


