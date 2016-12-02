%% Multidimensional scaling
% Plot membership against votes

clear
figure(1); clf

%% load data
%votes = load('votes.dat'); 

% load senetors
senator_MD = loadjson('senator_metadata.json');
senator_MD = struct2cell(senator_MD); 

% load membership
 Mship = load('membership.dat'); 
 Mship(isnan(Mship)) = 0; % set NaNs to 0

% count the number of times i is in same caucus as j
% count the number of times i voted the same as j

CoCauc = zeros(448,448); 
for i = 1:448
    for j = 1:448
        
        u1 = Mship(i,:); 
        u2 = Mship(j,:); 
        
        K(i,j) = exp( -.0001* norm(u1 - u2)^2) ; 
         
        
        u1(u1~=1 & u1~=0) == NaN; 
        u2(u2~=1 & u2~=0) == NaN; 
        
        CoCauc(i,j) = - 2*sum(u1==u2) ...
                 + sum(u1==u1) ...
                 + sum(u2==u2) ; 
end
end



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

    
%     if membership(i,31) == 1
%         Caucus1(i) = 1; 
%     elseif membership(i,145) == 1
%         Caucus1(i) = 2;
%     elseif membership(i, 270) == 1 
%         Caucus1(i) = 3;
%     elseif membership(i,200) == 1
%         Caucus1(i) = 4; 
%     elseif membership(i,339) == 1
%         Caucus1(i) = 5; 
%     elseif membership(i, 181) == 1
%         Caucus1(i) = 6;
%     else 
%         Caucus1(i) = 7; 
%     end 
        
end

%% MDS
figure(1); clf

Y = mdscale(CoCauc,2); 

scatter(Y(Party=='D',1), Y(Party=='D',2),'b', 'filled')
hold on
scatter(Y(Party=='R',1), Y(Party=='R',2),'r', 'filled')
hold off

title('Multidimensional scaling on roll call votes')
xlabel('x1')
ylabel('x2')
legend('Democrat', 'Repbulican')

%% Kernel pca

[U E V] = svd(K); 

figure(2); clf
scatter(U(Party=='D',1), U(Party=='D',2), 'filled')
hold on
scatter(U(Party=='R',1), U(Party=='R',2), 'filled')
hold off



