function [g, H] = derive(x, A0, A1)

  F = eye(3) + A0*x(1) + A1*x(2);
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
end

