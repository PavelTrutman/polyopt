% Centre analytique
% selon [Nesterov 2004] section 4.2.5, methode de Newton amortie (4.2.25)
% D. Henrion, 30 mars 2015

% LMI
%F0 = [1 0 0;0 1 0;0 0 2];
%F1 = [1 0 0;0 -1 0;0 0 -1];
%F2 = [0 1 0;1 0 1;0 1 0];
F0 = eye(3);
F1 = [1 1 0;
      1 1 0;
      0 0 0];
F2 = [1 0 1;
      0 0 1;
      1 1 1];

% represente le spectraedre avec YALMIP
%y=sdpvar(2,1);
%plot(F0+F1*y(1)+F2*y(2)>=0)
hold on

% choix du point initial
disp('cliquez dans le spectraedre');
%x = ginput(1)';
x = [0; 0];
X = x;

% calcul du centre analytique
lambda = 1;
while lambda > 1e-6
 F = F0+F1*x(1)+F2*x(2);
 Fi = inv(F);
 Fi1 = F1*Fi; Fi2 = F2*Fi;
 g1 = -trace(Fi1);
 g2 = -trace(Fi2);
 g = [g1;g2]; % gradient
 h11 = trace(Fi1^2);
 h21 = trace(Fi1*Fi2);
 h22 = trace(Fi2^2);
 H = [h11 h21; h21 h22]; % hessienne
 d = H\g; % direction de Newton
 lambda = sqrt(g'*d); % amortissement
 x = x - d/(1+lambda); % pas de Newton
 X = [X x]; % store the iterates
end

for k = 1:size(X,2)
 plot(X(1,k),X(2,k),'*k');
end


