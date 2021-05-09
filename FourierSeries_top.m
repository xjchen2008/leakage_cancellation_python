clc;clear;close all;
N = 100
n = linspace(1,N,1);
T0=2*pi;
T1=1.5*pi;
w0 = 2*pi/T0;
w1 = 2*pi/T1;
an =4*(-1)^n *cos((T0-T1)*pi)./(T0*(n.^2*w0^2-w1^2));
fourierBas