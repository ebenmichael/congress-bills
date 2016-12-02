%% Neigborhood regession on roll call votes

%% load data
votes = load('votes.dat'); 

% remove dead ppl (senators w multiple -1s) 
for i = 1 : size(votes, 2)
     if sum(votes(:,i)==-1) == 0; 
         keep(i) = logical(1); 
     end
end

votes = votes(:,keep); 
N = size(votes,2); % Number of senetors

% load senetors metadata
senator_MD = loadjson('senator_metadata.json');
senator_MD = struct2cell(senator_MD); 

% load membership
Membership = load('membership.dat'); 
%Membership(isnan(Membership)) = 0; % set NaNs to 0
Membership = Membership(keep,:); 


% set yes to +1; no to -1; other (absent, abstaining, etc) to 0
V = votes;
V(V==0) = -5;
V(V~= 1 & V~= -5) = 0;
V(V==-5) = -1;

Beta = zeros(N, N); 

for i = 1:N
    Y = V(:,i); 
    retain = (Y==1 | Y==-1); 
    Y = Y(retain); 
    Y(Y==-1) = 0; 
    
    X = votes(retain, :); 
    X(:,i) = []; 
    
    % 5-fold cross validation on the first regression
    if i == 1
        [B Fit] = lassoglm(X,Y, 'binomial', 'CV', 5, 'NumLambda', 50); 
    end
    
    
    B = lassoglm(X,Y, 'binomial', 'lambda', Fit.LambdaMinDeviance); 
        % lambda is sparsity parameter; 0.0067 for now, after 5-fold CV 
        % on first column, NumLambda = 5
   Beta(1:N~=i,i) = B;      
end
% 
% Beta2 = zeros(N,N); 
% 
% for i = 1:N 
%     Beta2(1:N~=i,i) = Beta(:,i); 
% end 
% 
% Beta = Beta2; 
% 


% create adjacency matrix A for graph
A = ones(N,N); 
for i = 1:N
    for j = 1:N-1
        
        A(i,j) = (Beta(i,j) ~= 0) | (Beta(i,j) ~= 0); 
        A(j,i) = A(i,j);   
        
    end
end


G = graph(A); 
figure(1); clf
plot(G)
title('House connectivity')

%% extract members of a caucus
    % caucuses (from liberal to conservative): 
    % Congressional Progressive Caucus: 270
    % New Democrat Coalition: 145
    % Blue Dog Coalition: 31
    % Tuesday Group: 200
    % Main street partnership: 339
    % Republical Study Committee: 181

    % COngr Black Caucus: 243
    % Congr Rural Caucus: 186
index = Membership(:,186)==1; 
Cauc_name = 'Congressional Rural Caucus'; 

A_sub = A(index, index); 
G_sub = graph(A_sub); 
figure(2); clf
plot(G_sub)
title(Cauc_name)


CaucusSize = sum(index)
HouseConnectivity = .5 * sum(A(:))/(N*(N-1))
CaucusConnectivity = .5 * sum(A_sub(:))/...
    (CaucusSize * (CaucusSize-1))

dim = [.2 .5 .3 .3];
str = sprintf('House connectivity: %f\n Caucus Connectivity: %f', ...
    HouseConnectivity, CaucusConnectivity);
annotation('textbox',dim,'String',str,'FitBoxToText','on');



% extract members of a party 
% for i = 1:length(senator_MD)
%     if senator_MD{i}.party == 'D'
%         Party(i) = 'D'; 
%     end
%     
%     if senator_MD{i}.party == 'R'
%         Party(i) = 'R'; 
%     end 
% end
% 
% Party = Party(keep); 
% 
% index = Party =='R'; 
% P_sub = A(index, index); 
% G_sub = graph(P_sub); 
% figure(3); clf
% plot(G_sub)
% 
% 
% PartySize = sum(index)
% theta = linspace(0, 2*pi, PartySize);  
% Coord = [cos(theta'), sin(theta')]; 
% 
% 
% figure(4); clf
% gplot(P_sub, Coord); 
% 
% HouseConnectivity = .5 * sum(A(:))/(N*(N-1))
% PartyConnectivity = .5 * sum(P_sub(:))/...
% (PartySize * (PartySize-1))





