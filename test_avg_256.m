clc;
close all;
clear;
%%%%%%%%%%%%%%%%
c = 3e8;
fs = 5e6;
N = 2560;
NFFT = 10*N;
tc = N/fs; %[sample/samples/sec]
K = fs / tc  % chirp rate
freq = (-fs/2: fs/N: fs/2-fs/N); %original code
freq = freq';
distance1 = c * freq / K / 2.0;
distance2 = c ./ freq / 2.0;
offset = 4*N;%Offset by number of bytes
t = linspace(0,tc,N);
pulse_to_avg=2; % this is how many pulses you want to get averaged
fileID_rx = fopen('usrp_samples_loopback.dat');
pulse_after_avg=1;


rx0_sum=zeros(pulse_after_avg,N);
w_HM = hamming(N);
w_BM = blackman(N);
w = w_BM;
%{
load refsig_B200_08142018_avg50_400000points_anttena.mat;
%load refsig_B200_08142018_avg100_400000points_realsig.mat;
%load refsig_B200_08142018_avg20_400000points.mat;
tx_sig=refsig_avg;
%plot(real(tx_sig));
TX = fft(tx_sig);  
%}
extra = 0;
xq = 1:0.5:N+0.5;
for n= 1:pulse_after_avg;n
    for m = 1:1:pulse_to_avg;m
        fseek(fileID_rx, offset*((n-1)*pulse_to_avg+(m-1)), 'bof');
        data_rx = fread(fileID_rx,2*(N+extra),'int16');
        rx0 = ((data_rx(2:2:end)) + 1i*(data_rx(1:2:end)))';
        rx0_sum(n,:)=rx0_sum(n,:)+rx0;
        %plot(real(RX0),'*');   
        %hold on;
    end
end
rx0_avg=conj(rx0_sum)/pulse_to_avg; % For some reason, IQ channals are inversed, so using conj
%plot(real(rx0_avg));
refsig_avg=rx0_avg;
plot(real(rx0_avg))
hold on;
plot(imag(rx0_avg));
%{
for m = 1:1:pulse_after_avg;m
    %RX = fft(rx0_avg(m,:).*w');
    %pc(:,(m))= ifft(conj(TX).*RX);
    %pc(:,(m))= ifft(conj(RX).*RX); % Auto correltation
    pc_mixer(:,m)=fft(conj(tx_sig).*rx0_avg.*w');
    
end

figure;
pc_mixer_log = fftshift(20*log10(abs(pc_mixer(:,m))));
plot(distance1, pc_mixer_log-max(pc_mixer_log),'.-');
xlim([0,1000]);
grid on;
%}
