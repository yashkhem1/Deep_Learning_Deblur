tic;
close all;

%% Wiener Filter
I = im2double(imread('motion-blur-result.jpg'));
% I = imresize(I, 0.5);
figure;
imshow(I);

trans = 20;
theta = 2;

PSF = fspecial('motion', trans, theta);

res_img = deconvwnr(I, PSF, 0.1);

fft_img = fftshift(fft2(res_img));

fft_img(1:170, 315:324) = zeros(170, 10);
fft_img(190:360, 315:324) = zeros(171, 10);

log_fft = log(abs(fft_img) + 1);


figure;
imshow(res_img);

%% PSNR
img1 = im2double(imread('319.png'));
img2 = im2double(imread('testOutput.jpeg'));
pSnr = psnr(img1, img2);
sSim = ssim(img1, img2);
toc;