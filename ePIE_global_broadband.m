%% Streamlined ePIE code for reconstructing from experimental diffraction patterns
function [big_obj,aperture,fourier_error,initial_obj,initial_aperture] = ePIE_global_broadband(ePIE_inputs,varargin)
%varargin = {beta_ap, beta_obj, modeSuppression}
optional_args = {1 1 0,0}; %default values for optional parameters
nva = length(varargin);
optional_args(1:nva) = varargin;
[beta_obj, beta_ap, modeSuppression, probe_norm] = optional_args{:};
rng('shuffle','twister');
%% setup working and save directories

dir = pwd;
save_string = [ dir '/Results_ptychography/']; % Place to save results


%% essential inputs
diffpats = ePIE_inputs(1).Patterns;
positions = ePIE_inputs(1).Positions;
filename = ePIE_inputs(1).FileName;
pixel_size = ePIE_inputs(1).PixelSize;
big_obj = ePIE_inputs(1).InitialObj;
aperture_radius = ePIE_inputs(1).ApRadius;
aperture = ePIE_inputs(1).InitialAp;
iterations = ePIE_inputs(1).Iterations;
lambda = ePIE_inputs(1).lambda;
S = ePIE_inputs(1).S;
[~,job_ID] = system('echo $JOB_ID');
job_ID = job_ID(~isspace(job_ID));
nModes = length(pixel_size);
filename = strcat('reconstruction_',filename,'_',job_ID);
filename = strrep(filename,'__','_');
%% parameter inputs
if isfield(ePIE_inputs, 'saveOutput')
    saveOutput = ePIE_inputs(1).saveOutput;
else
    saveOutput = 1;
end
if isfield(ePIE_inputs, 'saveIntermediate')
    saveIntermediate = ePIE_inputs(1).saveIntermediate;
else
    saveIntermediate = 0;
end
if isfield(ePIE_inputs, 'GpuFlag')
    gpu = ePIE_inputs(1).GpuFlag;
else
    gpu = 0;
end
if isfield(ePIE_inputs, 'apComplexGuess')
    apComplexGuess = ePIE_inputs(1).apComplexGuess;
else apComplexGuess = 0;
end

if isfield(ePIE_inputs, 'probeMaskFlag')
    probeMaskFlag = ePIE_inputs(1).probeMaskFlag;
else probeMaskFlag = 0;
end

if isfield(ePIE_inputs, 'supportMaskFlag')
    supportMaskFlag = ePIE_inputs(1).supportMaskFlag;
else supportMaskFlag = 0;
end

if isfield(ePIE_inputs, 'averagingConstraint')
    averagingConstraint = ePIE_inputs(1).averagingConstraint;
else
    averagingConstraint = 0;
end
if isfield(ePIE_inputs, 'Posi')
    strongPosi = ePIE_inputs(1).Posi;
else
    strongPosi = 0;
end
if isfield(ePIE_inputs, 'Realness')
    realness = ePIE_inputs(1).Realness;
else
    realness = 0;
end
if isfield(ePIE_inputs, 'updateAp')
    updateAp = ePIE_inputs.updateAp;
else
    updateAp = 1;
end
if isfield(ePIE_inputs, 'miscNotes')
    miscNotes = ePIE_inputs.miscNotes;
else
    miscNotes = 'None';
end
%% === Reconstruction parameters frequently changed === %%
beta_pos = 0.9; % Beta for enforcing positivity
do_posi = 0;
update_aperture_itt = 0;
%%
fprintf('dataset = %s\n',ePIE_inputs.FileName);
fprintf('output filename = %s\n', filename);
fprintf('iterations = %d\n',iterations);
fprintf('beta object = %0.1f\n',beta_obj);
fprintf('beta probe = %0.1f\n',beta_ap);
fprintf('number of modes = %d\n',nModes);
fprintf('gpu flag = %d\n',gpu);
fprintf('averaging objects = %d\n',averagingConstraint);
fprintf('complex probe guess = %d\n',apComplexGuess);
fprintf('probe mask flag = %d\n',probeMaskFlag);
fprintf('support mask flag = %d\n',supportMaskFlag);
fprintf('probe normalization = %d\n',probe_norm);
fprintf('strong positivity = %d\n',strongPosi);
fprintf('realness enforced = %d\n',realness);
fprintf('updating probe = %d\n',updateAp);
fprintf('enforcing positivity = %d\n',do_posi);
fprintf('updating probe after iteration %d\n',update_aperture_itt);
fprintf('mode suppression = %d\n',modeSuppression);
fprintf('misc notes: %s\n', miscNotes);
clear ePIE_inputs
%% Define parameters from data and for reconstruction
for ii = 1:size(diffpats,3)
    diffpats(:,:,ii) = fftshift(sqrt(diffpats(:,:,ii)));
end
goodInds = diffpats(:,:,1) ~= -1; %assume missing center homogenous
[N1,N2,nApert] = size(diffpats); % Size of diffraction patterns
best_err = 100; % check to make sure saving reconstruction with best error
little_area = N1; % ROI of reconstruction to place back into big obj
little_cent = floor(little_area/2) + 1;
cropVec = (1:N1) - little_cent;
mcm = @makeCircleMask;
for m = 1:length(lambda)
    %% Get centre positions for cropping (should be a 2 by n vector)
    [pixelPositions, bigx, bigy] = convert_to_pixel_positions_testing5(positions,pixel_size(m),little_area);
    centrey = round(pixelPositions(:,2));
    centrex = round(pixelPositions(:,1));
    centBig = round((bigx+1)/2);
    Y1(:,m) = centrey - floor(N1/2); Y2(:,m) = Y1(:,m)+N1-1;
    X1(:,m) = centrex - floor(N1/2); X2(:,m) = X1(:,m)+N1-1;    
    for ii = 1:nApert
        cropR(ii,:,m) = cropVec+centBig+(centrey(ii)-centBig);
        cropC(ii,:,m) = cropVec+centBig+(centrex(ii)-centBig);
    end
    %% create initial aperture?and object guesses
    if aperture{m} == 0
        if apComplexGuess == 1
            aperture{m} = single(((feval(mcm,(ceil(aperture_radius./pixel_size(m))),little_area).*...
                rand(little_area,little_area) .* exp(1i*rand(little_area,little_area)))));
        else
            aperture{m} = single(feval(mcm,(ceil(aperture_radius./pixel_size(m))),little_area));
        end
        
        initial_aperture{m} = aperture{m};
    else
        %         display('using supplied aperture')
        aperture{m} = single(aperture{m});
        initial_aperture{m} = aperture{m};
    end
    
    if probeMaskFlag == 1
        %         display('applying loose support')
        %     probeMask{m} = double(aperture{m} > 0);
        probeMask{m} = double(feval(mcm,(ceil(aperture_radius./pixel_size(m))),little_area));
    else
        probeMask{m} = [];
    end
    
    if big_obj{m} == 0
        big_obj{m} = single(rand(bigx,bigy)).*exp(1i*(rand(bigx,bigy)));
        initial_obj{m} = big_obj{m};
    else
        big_obj{m} = single(big_obj{m});
        initial_obj{m} = big_obj{m};
    end
    
    if supportMaskFlag ==1
        %{
        radius = round(aperture_radius/pixel_size(m)*1.4);
        r = floor(bigx/2) - radius + (0:2*radius);
        support{m} = zeros(bigx,bigy);
        support{m}(r,r)=1;
        support{m} = ~support{m};
        %}
        support{m} = zeros(bigx,bigy);
        mask_obj= logical(aperture{m});
        for ii=1:nApert
            support{m}(Y1(ii,m):Y2(ii,m), X1(ii,m):X2(ii,m)) = support{m}(Y1(ii,m):Y2(ii,m), X1(ii,m):X2(ii,m)) | mask_obj;
        end
        support{m} = ~support{m};
    end    
end
fourier_error = zeros(iterations,nApert);

%% GPU
if gpu == 1
    display('========ePIE reconstructing with GPU========')
    diffpats = gpuArray(diffpats);
    fourier_error = gpuArray(fourier_error);
    big_obj = cellfun(@gpuArray, big_obj, 'UniformOutput', false);
    aperture = cellfun(@gpuArray, aperture, 'UniformOutput', false);
    S = gpuArray(S);
else
    display('========ePIE reconstructing with CPU========')
end
cdp = class(diffpats);

nMode=length(lambda);
big_obj_numer = cell(nMode);
big_obj_denom = cell(nMode);
probe_numer = cell(nMode);
probe_denom = cell(nMode);
abs_ap_2 = cell(nMode,1);
u_ii = cell(nMode,1);
z_ii = cell(nMode,1);
probe_max = zeros(nMode,1);
object_max = zeros(nMode,1);
%Pu = cell(nMode,1);
for m=1:nMode
    big_obj_numer{m} = zeros(size(big_obj{m}));
    big_obj_denom{m} = zeros(size(big_obj{m}));
    probe_numer{m} = zeros(little_area);
    probe_denom{m} = zeros(little_area);
end

%% Main ePIE itteration loop
disp('========beginning reconstruction=======');
for itt = 1:iterations
    tic
    for m=1:nMode
        abs_ap_2{m} =  abs(aperture{m}).^2;
        probe_max(m) = max(abs_ap_2{m}(:));
        big_obj_numer{m}(:) = 0;
        big_obj_denom{m}(:) = 0;
        probe_numer{m}(:) = 0;
        probe_denom{m}(:) = 0;
    end
    
    %% collect magnitudes
    for ii = randperm(nApert)
        current_dp = diffpats(:,:,ii);
        collected_mag = zeros(N1,N2,cdp);
        
        for m = 1:length(lambda)
            %rspace = big_obj{m}(cropR(aper,:,m), cropC(aper,:,m));
            u_ii{m} = big_obj{m}(Y1(ii,m):Y2(ii,m), X1(ii,m):X2(ii,m));
            object_max(m) = max(abs(u_ii{m}(:)))^2;            
            
            % Create new exitwave
            Pu = u_ii{m}.*aperture{m};
            z_ii{m} = fft2(Pu);            
            collected_mag = collected_mag + abs(z_ii{m}).^2;
        end
        collected_mag = sqrt(collected_mag);
        scale = current_dp./collected_mag;
        scale(isnan(scale))=0;
        
        for m = 1:length(lambda)
            z_ii{m}(goodInds) = scale(goodInds) .* z_ii{m}(goodInds);           
            Pu = ifft2(z_ii{m});

            r = Y1(ii,m):Y2(ii,m);
            c = X1(ii,m):X2(ii,m);  

            big_obj_numer{m}(r,c) = big_obj_numer{m}(r,c) + Pu.*conj(aperture{m});
            big_obj_denom{m}(r,c) = big_obj_denom{m}(r,c) + abs_ap_2{m};
            probe_numer{m} = probe_numer{m} + conj(u_ii{m}).*Pu;
            probe_denom{m} = probe_denom{m} + abs(u_ii{m}).^2;
        end
        
        % compute error
        fourier_error(itt,ii)=sum( abs(current_dp(goodInds)-collected_mag(goodInds)) )...
            ./ sum(current_dp(goodInds));
    end
    
    %% update big_obj{m} & probe{m}
    for m = 1:nMode
        % update big_obj{m}     
        big_obj{m} = ((1-beta_obj)*big_obj{m} + (beta_obj/probe_max(m)/nApert)* big_obj_numer{m}) ...
            ./ (1-beta_obj + (beta_obj/probe_max(m)/nApert)*big_obj_denom{m});
        
        if strongPosi == 1, big_obj{m}(big_obj{m} < 0) = 0; end
        if realness == 1,  big_obj{m} = real(big_obj{m}); end
        if supportMaskFlag, big_obj{m}(support{m})=0; end
        
        % update probe{m}
        if itt > update_aperture_itt && updateAp == 1
            if modeSuppression==0 || mod(m,3)~=0
                aperture{m} = ((1-beta_ap)*aperture{m} + (beta_ap/object_max(m)/nApert)*probe_numer{m} )./ (1-beta_ap + (beta_ap/object_max(m)/nApert)* probe_denom{m});
                %probe normalization
                if probe_norm, aperture{m} = aperture{m}/max(abs(aperture{m}(:))); end
            end
            if probeMaskFlag==1, aperture{m}=aperture{m}.*probeMask{m}; end
        end        
    end
    
    %% plot result
    for m=1:length(lambda)
        %[dim1,dim2] = size(big_obj{m});
        %r = floor(dim1/2)+ (-64:64); c = floor(dim2/2)+ (-64:64);
        figure(m); 
        subplot(1,2,1); imagesc(abs(big_obj{m})); colormap jet; axis image;
        subplot(1,2,2); imagesc(abs(aperture{m})); colormap jet; axis image;
        drawnow;
    end    
    
    %% averaging between wavelengths
    if averagingConstraint == 1
        %         if gpu == 1
        averaged_obj = zeros([size(big_obj{1}) length(lambda)], cdp);
        interpMethod = 'linear';
        %         else
        %             averaged_obj = zeros([size(big_obj{1}) length(lambda)]);
        %             interpMethod = 'linear';
        %         end
        
        first_obj = big_obj{1};
        averaged_obj(:,:,1) = first_obj;
        ndim = floor(size(big_obj{1},1)/2);
        [xm, ym] = meshgrid(-ndim:ndim, -ndim:ndim);
        %k_arr = zeros(1,length(lambda));
        %k_arr(1) = 1;
        %rescaling all the objects to have the same pixel size as first obj
        %         parfor m = 2:length(lambda)
        for m = 2:length(lambda)
            xm_rescaled = xm .* (pixel_size(m) / pixel_size(1));
            ym_rescaled = ym .* (pixel_size(m) / pixel_size(1));
            ctrPixel = ceil((size(big_obj{m},1)+1) / 2);
            cropROI = big_obj{m}(ctrPixel-ndim:ctrPixel+ndim, ctrPixel-ndim:ctrPixel+ndim);
            resized_obj = interp2(xm_rescaled, ym_rescaled, cropROI, xm, ym, interpMethod, 0);
            resized_obj(resized_obj < 0) = 0;
            %no normalization for now
            %k_arr(m) = normalizer(first_obj, resized_obj);
            averaged_obj(:,:,m) = resized_obj;
        end
        averaged_obj = sum(averaged_obj,3) ./ length(lambda);
        %distribute back to big_objs
        big_obj{1} = averaged_obj;
        %         parfor m = 2:length(lambda)
        for m = 2:length(lambda)
            xm_rescaled = xm .* (pixel_size(1) / pixel_size(m));
            ym_rescaled = ym .* (pixel_size(1) / pixel_size(m));
            resized_obj = interp2(xm_rescaled, ym_rescaled, averaged_obj, xm, ym, interpMethod, 0);
            resized_obj(resized_obj < 0) = 0;
            ctrPixel = ceil((size(big_obj{m},1)+1) / 2);
            big_obj{m}(ctrPixel-ndim:ctrPixel+ndim, ctrPixel-ndim:ctrPixel+ndim) = resized_obj;           
        end
    end
       
    mean_err = sum(fourier_error(itt,:),2)/nApert;
    
    if best_err > mean_err
        for m = 1:length(lambda)
            best_obj{m} = big_obj{m};
        end
        best_err = mean_err;
    end
    if saveOutput == 1
        if itt == 50 && saveIntermediate == 1
            if gpu == 1
                best_obj = cellfun(@gather, best_obj, 'UniformOutput', false);
            end
            save([filename '_iter_' num2str(itt) '.mat'], 'best_obj', '-v7.3');
            best_obj = cellfun(@gpuArray, best_obj, 'UniformOutput', false);
        end
    end
    toc
    fprintf('%d. Error = %f\n',itt,mean_err);
end
disp('======reconstruction finished=======')


if gpu == 1
    fourier_error = gather(fourier_error);
    best_obj = cellfun(@gather, best_obj, 'UniformOutput', false);
    aperture = cellfun(@gather, aperture, 'UniformOutput', false);
    big_obj = cellfun(@gather, big_obj, 'UniformOutput', false);
    initial_aperture = cellfun(@gather, initial_aperture, 'UniformOutput', false);
    % S = cellfun(@gather, S, 'UniformOutput', false);
    S = gather(S);
end

if saveOutput == 1
    save([save_string filename '.mat'],'best_obj','aperture','big_obj','initial_aperture','fourier_error','S');
end

%% Function for converting positions from experimental geometry to pixel geometry

    function [positions, bigx, bigy] = convert_to_pixel_positions(positions,pixel_size,little_area)
        positions = positions./pixel_size;
        positions(:,1) = (positions(:,1)-min(positions(:,1)));
        positions(:,2) = (positions(:,2)-min(positions(:,2)));
        positions(:,1) = (positions(:,1)-round(max(positions(:,1))/2));
        positions(:,2) = (positions(:,2)-round(max(positions(:,2))/2));
        positions = round(positions);
        bigx =little_area + max(positions(:))*2+10; % Field of view for full object
        bigy = little_area + max(positions(:))*2+10;
        big_cent = floor(bigx/2)+1;
        positions = positions+big_cent;
        
        
    end


    function [pixelPositions, bigx, bigy] = ...
            convert_to_pixel_positions_testing5(positions,pixel_size,little_area)
        
        
        pixelPositions = positions./pixel_size;
        pixelPositions(:,1) = (pixelPositions(:,1)-min(pixelPositions(:,1))); %x goes from 0 to max
        pixelPositions(:,2) = (pixelPositions(:,2)-min(pixelPositions(:,2))); %y goes from 0 to max
        pixelPositions(:,1) = (pixelPositions(:,1) - round(max(pixelPositions(:,1))/2)); %x is centrosymmetric around 0
        pixelPositions(:,2) = (pixelPositions(:,2) - round(max(pixelPositions(:,2))/2)); %y is centrosymmetric around 0
        
        bigx = little_area + round(max(pixelPositions(:)))*2+10; % Field of view for full object
        bigy = little_area + round(max(pixelPositions(:)))*2+10;
        
        big_cent = floor(bigx/2)+1;
        
        pixelPositions = pixelPositions + big_cent;
        
    end

%% 2D guassian smoothing of an image

    function [smoothImg,cutoffRad]= smooth2d(img,resolutionCutoff)
        
        Rsize = size(img,1);
        Csize = size(img,2);
        Rcenter = round((Rsize+1)/2);
        Ccenter = round((Csize+1)/2);
        a=1:1:Rsize;
        b=1:1:Csize;
        [bb,aa]=meshgrid(b,a);
        sigma=(Rsize*resolutionCutoff)/(2*sqrt(2));
        kfilter=exp( -( ( ((sqrt((aa-Rcenter).^2+(bb-Ccenter).^2)).^2) ) ./ (2* sigma.^2) ));
        kfilter=kfilter/max(max(kfilter));
        kbinned = my_fft(img);
        
        kbinned = kbinned.*kfilter;
        smoothImg = my_ifft(kbinned);
        
        [Y, X] = ind2sub(size(img),find(kfilter<(exp(-1))));
        
        Y = Y-(size(img,2)/2);
        X = X-(size(img,2)/2);
        R = sqrt(Y.^2+X.^2);
        cutoffRad = ceil(min(abs(R)));
    end

%% Fresnel propogation
    function U = fresnel_advance (U0, dx, dy, z, lambda)
        % The function receives a field U0 at wavelength lambda
        % and returns the field U after distance z, using the Fresnel
        % approximation. dx, dy, are spatial resolution.
        
        k=2*pi/lambda;
        [ny, nx] = size(U0);
        
        Lx = dx * nx;
        Ly = dy * ny;
        
        dfx = 1./Lx;
        dfy = 1./Ly;
        
        u = ones(nx,1)*((1:nx)-nx/2)*dfx;
        v = ((1:ny)-ny/2)'*ones(1,ny)*dfy;
        
        O = my_fft(U0);
        
        H = exp(1i*k*z).*exp(-1i*pi*lambda*z*(u.^2+v.^2));
        
        U = my_ifft(O.*H);
    end

%% Make a circle of defined radius

    function out = makeCircleMask(radius,imgSize)
        
        
        nc = imgSize/2+1;
        n2 = nc-1;
        [xx, yy] = meshgrid(-n2:n2-1,-n2:n2-1);
        R = sqrt(xx.^2 + yy.^2);
        out = R<=radius;
    end

%% Function for creating HSV display objects for showing phase and magnitude
%  of a reconstruction simaultaneously

    function [hsv_obj] = make_hsv(initial_obj, factor)
        
        [sizey,sizex] = size(initial_obj);
        hue = angle(initial_obj);
        
        value = abs(initial_obj);
        hue = hue - min(hue(:));
        if sum(hue(:)) == 0
            
        else
            hue = (hue./max(hue(:)));
        end
        value = (value./max(value(:))).*factor;
        hsv_obj(:,:,1) = hue;
        hsv_obj(:,:,3) = value;
        hsv_obj(:,:,2) = ones(sizey,sizex);
        hsv_obj = hsv2rgb(hsv_obj);
    end
%% Function for defining a specific region of an image

    function [roi, bigy, bigx] = get_roi(image, centrex,centrey,crop_size)
        
        bigy = size(image,1);
        bigx = size(image,2);
        
        half_crop_size = floor(crop_size/2);
        if mod(crop_size,2) == 0
            roi = {centrex - half_crop_size:centrex + (half_crop_size - 1);...
                centrey - half_crop_size:centrey + (half_crop_size - 1)};
            
        else
            roi = {centrex - half_crop_size:centrex + (half_crop_size);...
                centrey - half_crop_size:centrey + (half_crop_size)};
            
        end
    end

%% Fast Fourier transform function
    function kspace = my_fft(rspace)
        %MY_FFT computes the FFT of an image
        %
        %   last modified 1/12/17
        
        kspace = fftshift(fftn(rspace));
    end
%% Inverse Fast Fourier transform function
    function rspace = my_ifft(kspace)
        %MY_IFFT computes the IFFT of an image
        %
        %   last modified 1/12/17
        
        rspace = ifftn(ifftshift(kspace));
    end
end


