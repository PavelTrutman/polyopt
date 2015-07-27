% SDP solver
% Pavel Trutman, pavel.trutman@fel.cvut.cz

% min c_0*x_0 + c_1*x_1
%  s. t. I_3 + A_0*x_0 + A_1*x_1 >= 0 (semidefinite positive)

% initialization of the problem
c = [3; -2];
A0 = [1 1 0;
      1 1 0;
      0 0 0];
A1 = [1 0 1;
      0 0 1;
      1 1 1];
nu = 3;

% some constants
beta = 1/9;
gamma = 5/36;


% Auxiliary path-following scheme
t = 1;
k = 0;

% starting point
y = [0; 0];

% gradient and hessian
[g, H] = derive(y, A0, A1);

gy0 = g;

% iteration process
while true
  k = k + 1
  t = t - gamma/sqrt((H\gy0)'*gy0)
  y = y - H\(t*gy0 + g)
  
  % gradient and hessian
  [g, H] = derive(y, A0, A1);
  
  if sqrt((H*g)'*g) <= sqrt(beta)/(1 + sqrt(beta))
    % break if the stoping condition is met
    break;
  end
  
end

% prepare x
x = y - H\g

% Main path-following scheme
t = 0;
eps = 10^(-3);
k = 0;

% gradient and hessian
[g, H] = derive(x, A0, A1);

% initial condition
initCondition = sqrt((H\g)'*g)

% iteration process
while true
  k = k + 1
  
  % gradient and hessian
  [g, H] = derive(x, A0, A1);

  t = t + gamma/sqrt((H\c)'*c)
  x = x - H\(t*c+g)
  
  if eps*t >= nu + (beta + sqrt(nu))*beta/(1 - beta)
    % break if the stoping condition is met
    break;
  end
  
end