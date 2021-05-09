function FourierSeries(n,xn,T0)
% Plot signal from Fourier Series
% Inputs:
% n: indices for Fourier series coefficients
% xn: Fourier series coefficients (complex)
% T0: Signal period

if length(n) ~= length(xn)
    error('The length of n and xn must match')
end
t = linspace(-2*T0,2*T0,1000);
x = zeros(size(t));
for ii=1:length(n)
    x = x+xn(ii)*exp(1i*n(ii)*2*pi*t/T0);
end
x = real(x);
figure,subplot(311)
stem(n,abs(xn)),xlabel('n'),ylabel('|x_n|')
subplot(312)
stem(n,angle(xn)),xlabel('n'),ylabel('\angle x_n')
subplot(313)
plot(t,x),xlabel('t'),ylabel('x(t)')