A0 = [1 1 0;
      1 1 0;
      0 0 0];
A1 = [1 0 1;
      0 0 1;
      1 1 1];
nu = 3;

beta = 1/9;
gamma = 5/36;


% AUX
t = 1;
k = 0;

y = [0; 0];

F = eye(3) + A0*y(1) + A1*y(2);
Fi = inv(F);
Fi0 = A0*Fi;
Fi1 = A1*Fi;
g0 = -trace(Fi0);
g1 = -trace(Fi1);
g = [g0; g1];
h00 = trace(Fi0^2);
h11 = trace(Fi1^2);
h01 = trace(Fi0*Fi1);
H = [h00 h01; h01 h11];

gy0 = g;

while true
  k = k + 1
  g
  H
  t = t - gamma/sqrt((H\gy0)'*gy0)
  y = y - H\(t*gy0 + g)
  
  F = eye(3) + A0*y(1) + A1*y(2);
  Fi = inv(F);
  Fi0 = A0*Fi;
  Fi1 = A1*Fi;
  g0 = -trace(Fi0);
  g1 = -trace(Fi1);
  g = [g0; g1];
  h00 = trace(Fi0^2);
  h11 = trace(Fi1^2);
  h01 = trace(Fi0*Fi1);
  H = [h00 h01; h01 h11];
  
  if sqrt((H*g)'*g) <= sqrt(beta)/(1 + sqrt(beta))
    break;
  end
  
end

