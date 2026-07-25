# Method

Use a fixed countable parameter/data space embedded in R, a deterministic
correctly specified simulator `x_i=theta`, and a fixed three-value summary
space. Under the true theta=0, the original summary posterior is supported on
`{0,1/N}` and converges weakly to delta_0.

The bounded continuous ISPD kernel is
`kappa(x,y)=1+g(x) exp(-(x-y)^2) g(y)` with
`g(x)=x^2 exp(-x^2)`. Its MMD sends delta_N toward delta_0 even though delta_N
does not converge weakly. The exact MDS objective selects the summary whose
posterior is delta_N for every N>=3. An independent checker reconstructs every
objective and a Gaussian `C_0` kernel is the negative control.
