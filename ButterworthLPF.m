function ButterworthLPF(omegac,N)
% Plots Butterworth LPF magnitude response
% Inputs:
% omegac: Half-power frequency, radians/second
% N: number of poles/filter order
% Poles
p = omegac*exp(1i*pi*(2*(1:N)+N-1)/(2*N));
omega = 0:5000;
H = ((1i*omegac)^4)*ones(size(omega));
for k=1:N
    H = H./(1i*2*pi*omega-p(k));
end
figure,plot(omega,abs(H))
xlabel('\omega/2\pi'),ylabel('|H(\omega)|')