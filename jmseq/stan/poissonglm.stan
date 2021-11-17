
data {
  int < lower = 1 > N; // number of person-time intervals
  int < lower = 1 > P; // number of columns in X matrix
  int < lower=0, upper=1 > y[N]; // event status
  vector[N] logtobs; // offset term calculated as log interval length
  matrix[N, P] X; // X matrix with no intercept
}

transformed data {
  matrix[N, P] Q_ast;
  matrix[P, P] R_ast;
  matrix[P, P] R_ast_inverse;
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(X)[, 1:P] * sqrt(N - 1);
  R_ast = qr_thin_R(X)[1:P, ] / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}

parameters {
  real beta0;               // intercept for log hazard rate
  vector[P] theta;           // coefficients on Q_ast 
 }

transformed parameters {
  vector[N] logmu = logtobs + beta0 + Q_ast * theta;
 }

model {
  target += poisson_log_lpmf(y | logmu); // y * logmu - exp(logmu)
}

generated quantities {
  vector[P] beta;
  beta = R_ast_inverse * theta; // coefficients on X
}

