load 'ePIE_inputs_20180226-SCF_USAF_laser_2.mat'
%https://drive.google.com/open?id=1p44LSrTQlbAwU2h2Y3W6906fxMBBCPQs
npats = size(ePIE_inputs.Positions,1);
index = randperm(npats,round(npats*.5));


%%
ePIE_inputs.FileName = 'DR_test';
ePIE_inputs.GpuFlag = 0;
ePIE_inputs.Patterns = ePIE_inputs.Patterns(:,:,index);
ePIE_inputs.Positions = ePIE_inputs.Positions(index,:);
ePIE_inputs.updateAp = 1;
ePIE_inputs.showim = 1;
%%
ePIE_inputs.Iterations = 300;
tic
%[beta_obj, beta_ap, probeNorm, init_weight, final_weight, order, semi_implicit_P]
% ePIE
%[big_obj,aperture,fourier_error,initial_obj,initial_aperture] = ePIE(ePIE_inputs,1,0.9);

% rPIE
%[big_obj2,aperture2,fourier_error2,initial_obj2,initial_aperture2] = rPIE(ePIE_inputs,0.1,1);

% DR
[big_obj3,aperture3,fourier_error3,initial_obj3,initial_aperture3] = DRb(ePIE_inputs, 0.9, 0.5, 0.9);
% (beta_obj, beta_ap, momentum) = (0.9,0.5.0.9) for aggressive but it might
% diverse
% (beta_obj, beta_ap, momentum) = (0.5,0.1.0.1) is safer if noise level is
% high
toc;
%%
[size1,size2] = size(big_obj);
half1 = floor(size1/2);
w = 135;
c1 = half1-w+1; c2 = half1+w;
figure(11); img(big_obj (c1:c2,c1:c2),'caxis',[0,1],'colormap','gray');
figure(21); img(big_obj2(c1:c2,c1:c2),'colormap','gray');
figure(31); img(big_obj3(c1:c2,c1:c2),'colormap','gray');



