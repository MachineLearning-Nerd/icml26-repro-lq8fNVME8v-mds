# Method

Use S=[-1/2,1/2], decoder P(x|s)=Normal(s,1), Gaussian kernel
`exp(-(x-y)^2/2)`, and
Q=(delta_{-sqrt(2)}+delta_{sqrt(2)})/2. The Q-objective has a unique but
quartically flat minimum at s=0. Contamination at y=sqrt(2) therefore gives
`s*(Q_epsilon) ~ (6 sqrt(2) epsilon)^(1/3)`.

The posterior density is proportional to `exp(-(theta-s)^2)` on [-1,1].
Consequently KL(P_0,P_s) is quadratic in s and KL/epsilon diverges as
epsilon^(-1/3). The verifier audits each printed assumption, performs an
80-decimal calibrated epsilon sweep, and checks the independently derived
constants. A regular correctly specified Q is the negative control.
