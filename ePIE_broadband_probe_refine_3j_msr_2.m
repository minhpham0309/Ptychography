%% jianhua method for higher energies preallocated fresnel propagators and not applying probe refinement every microiteration for speed
function [big_obj,aperture,fourier_error,initial_obj,initial_aperture] = ePIE_broadband_probe_refine_3j_msr_2(ePIE_inputs,varargin)
%varargin = {beta_ap, beta_obj, modeSuppression}
optional_args = {1 1 0}; %default values for optional parameters
nva = length(varargin);
optional_args(1:nva) = varargin;
[beta_ap, beta_obj, modeSuppression] = optional_args{:};
rng('shuffle','twister');
%% setup working and save directories

dir = pwd;
save_string = [ dir '/Results_ptychography/']; % Place to save results


%% essential inputs
diffpats = ePIE_inputs(1).Patterns; 
positions = ePIE_inputs(1).Positions;
filename = ePIE_inputs(1).FileName;
pixel_size = ePIE_inputs(1).PixelSize;
pixel_size_fresnel = ePIE_inputs(1).pixel_size_fresnel;
big_obj = ePIE_inputs(1).InitialObj;
aperture_radius = ePIE_inputs(1).ApRadius;
aperture = ePIE_inputs(1).InitialAp;
iterations = ePIE_inputs(1).Iterations;
lambda = ePIE_inputs(1).lambda;
S = ePIE_inputs(1).S;
[~,job_ID] = system('echo $JOB_ID');
job_ID = job_ID(~isspace(job_ID));
nModes = length(pixel_size);
central_mode = ePIE_inputs.central_mode; %best mode for probe replacement
fresnel_dist = ePIE_inputs.fresnel_dist; %probe to sample
filename = strcat('reconstruction_probe_replace_3j_msr_2_',filename,'_',job_ID);
filename = strrep(filename,'__','_');
%% parameter inputs
if isfield(ePIE_inputs, 'saveOutput')
    saveOutput = ePIE_inputs(1).saveOutput;
else
    saveOutput = 1;
end
if isfield(ePIE_inputs, 'save_intermediate');
    save_intermediate = ePIE_inputs.save_intermediate;
else
    save_intermediate = 1;
end
if isfield(ePIE_inputs, 'GpuFlag')
    gpu = ePIE_inputs(1).GpuFlag;
else
    gpu = 0;
end
if isfield(ePIE_inputs, 'apComplexGuess')
    apComplexGuess = ePIE_inputs(1).apComplexGuess;
else
    apComplexGuess = 0;
end
if isfield(ePIE_inputs, 'probeMaskFlag')
    probeMaskFlag = ePIE_inputs(1).probeMaskFlag;
else
    probeMaskFlag = 0;
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
if isfield(ePIE_inputs, 'update_aperture_after');
    update_aperture_after = ePIE_inputs.update_aperture_after;
else
    update_aperture_after = 0;
end
if isfield(ePIE_inputs, 'refine_aperture_after');
    refine_aperture_after = ePIE_inputs.refine_aperture_after;
else
    refine_aperture_after = 0;
end
if isfield(ePIE_inputs, 'refine_aperture_until');
    refine_aperture_until = ePIE_inputs.refine_aperture_until;
else
    refine_aperture_until = Inf;
end
if isfield(ePIE_inputs, 'probe_repl_freq')
    probe_repl_freq = ePIE_inputs(1).probe_repl_freq;
else
    probe_repl_freq = 10; %10% replacement frequency
end
if isfield(ePIE_inputs, 'miscNotes')
    miscNotes = ePIE_inputs.miscNotes;
else
    miscNotes = 'shuffling positions every iteration';
end
%% === Reconstruction parameters frequently changed === %%
beta_pos = 0.9; % Beta for enforcing positivity
do_posi = 0;
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
fprintf('strong positivity = %d\n',strongPosi);
fprintf('realness enforced = %d\n',realness);
fprintf('updating probe = %d\n',updateAp);
fprintf('enforcing positivity = %d\n',do_posi);
fprintf('updating probe after iteration %d\n',update_aperture_after);
fprintf('refining probe after iteration %d\n',refine_aperture_after);
fprintf('refining probe until iteration %d\n',refine_aperture_until);
fprintf('mode suppression = %d\n',modeSuppression);
fprintf('fresnel distance = %f\n',fresnel_dist);
fprintf('central mode = %d\n',central_mode);
fprintf('save intermediate = %d\n', save_intermediate);
fprintf('probe replacement frequency = %f\n',probe_repl_freq);
fprintf('misc notes: %s\n', miscNotes);
clear ePIE_inputs
%% Define parameters from data and for reconstruction
for ii = 1:size(diffpats,3)
    diffpats(:,:,ii) = single(fftshift(diffpats(:,:,ii)));
end
goodInds = find(diffpats(:,:,1) ~= -1); %assume missing center homogenous
[little_area,~] = size(diffpats(:,:,1)); % Size of diffraction patterns
nApert = size(diffpats,3);
best_err = 100; % check to make sure saving reconstruction with best error
little_cent = floor(little_area/2) + 1;
cropVec = (1:little_area) - little_cent;
mcm = @makeCircleMask;
for m = 1:length(lambda)
%% Get centre positions for cropping (should be a 2 by n vector)
    [pixelPositions, bigx, bigy] = convert_to_pixel_positions_testing5(positions,pixel_size(m),little_area);
    centrey = round(pixelPositions(:,2));
    centrex = round(pixelPositions(:,1));
    centBig = round((bigx+1)/2);
    for aper = 1:nApert
        cropR(aper,:,m) = cropVec + centrey(aper);
        cropC(aper,:,m) = cropVec + centrex(aper);
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
    sub_ap{m} = 1e-4 .* single(rand(little_area,little_area)).*exp(1i*(rand(little_area,little_area)));
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
    sub_obj{m} = big_obj{m};
    if save_intermediate == 1
        inter_obj{m} = zeros([size(big_obj{m}) 10]);
        inter_frame = 0;
    else
        inter_obj = [];
    end
%     display(size(big_obj{m}));

end
S_sub = S;
fourier_error = zeros(iterations,nApert);
%% probe replacement parameters
% scaling_ratio = pixel_size ./ pixel_size(central_mode); 
scaling_ratio = max(pixel_size(central_mode)./pixel_size, pixel_size ./ pixel_size(central_mode));
for mm = 1:length(lambda)
    scoop_size = round(little_area/scaling_ratio(mm));
    scoop_center = round((scoop_size+1)/2);
    scoop_vec{mm} = (1:scoop_size) - scoop_center + little_cent;
    scoop_range(mm) = range(scoop_vec{mm})+1;
    if scoop_range(mm) > little_area
        pad_pre(mm) = ceil((scoop_range(mm)-little_area)/2);
        pad_post(mm) = floor((scoop_range(mm)-little_area)/2);
    else
        pad_pre(mm) = 0;
        pad_post(mm) = 0;
    end
end
% cutoff = floor(iterations/2);
% prb_rplmnt_weight = min((cutoff^4/10)./(1:iterations).^4,0.1);
prb_rplmnt_weight = 0.1 .* ones(iterations,1);
%% pre allocation of propagators
for mm = 1:length(lambda)
    k = 2*pi/lambda(mm);
    Lx = pixel_size_fresnel(mm)*little_area;
    Ly = pixel_size_fresnel(mm)*little_area;
    dfx = 1./Lx;
    dfy = 1./Ly;
    u = ones(little_area,1)*((1:little_area)-little_area/2)*dfx;
    v = ((1:little_area)-little_area/2)'*ones(1,little_area)*dfy;
    if mm ~= central_mode
        H_fwd{mm} = ifftshift(exp(1i*k*fresnel_dist).*exp(-1i*pi*lambda(mm)*fresnel_dist*(u.^2+v.^2)));
        H_bk{mm} = exp(1i*k*-fresnel_dist).*exp(-1i*pi*lambda(mm)*-fresnel_dist*(u.^2+v.^2));
    else
        %H_fwd{mm} =exp(1i*k*fresnel_dist).*exp(-1i*pi*lambda(mm)*fresnel_dist*(u.^2+v.^2));
        H_bk{mm} = exp(1i*k*-fresnel_dist).*exp(-1i*pi*lambda(mm)*-fresnel_dist*(u.^2+v.^2));
        H_bk_shifted = ifftshift(H_bk{mm});
    end
end

%% GPU
if gpu == 1
    display('========ePIE probe refine(method 3j) reconstructing with GPU========')
    diffpats = gpuArray(diffpats);
    fourier_error = gpuArray(fourier_error);
    big_obj = cellfun(@gpuArray, big_obj, 'UniformOutput', false);
    sub_obj = cellfun(@gpuArray, sub_obj, 'UniformOutput', false);
    aperture = cellfun(@gpuArray, aperture, 'UniformOutput', false);
    sub_ap = cellfun(@gpuArray, sub_ap, 'UniformOutput', false);
    S = gpuArray(S);
else
    display('========ePIE probe refine(method 3j) reconstructing with CPU========')
end
cdp = class(diffpats);

%% Main ePIE itteration loop
disp('========beginning reconstruction=======');
for itt = 1:iterations
    tic
    
    if mod(itt,probe_repl_freq) == 0 && itt >= refine_aperture_after && itt <= refine_aperture_until
        probe_refinement_flag = 1;
        fprintf('probe replacement on\n');
    else
        probe_refinement_flag = 0;
    end
    
    for aper = randperm(nApert) 
        current_dp = diffpats(:,:,aper);
        pos_shuffle = randperm(nApert);
        cropR_sub(pos_shuffle,:,:) = cropR(1:nApert,:,:);
        cropC_sub(pos_shuffle,:,:) = cropC(1:nApert,:,:);
        for m = 1:length(lambda)
            rspace = big_obj{m}(cropR(aper,:,m), cropC(aper,:,m));
            rspace_sub = sub_obj{m}(cropR_sub(aper,:,m), cropC_sub(aper,:,m));
            buffer_rspace{m} = rspace;
            buffer_rspace_sub{m} = rspace_sub;
            object_max{m} = max(abs(rspace(:)));
            object_max_sub{m} = max(abs(rspace_sub(:)));
            probe_max{m} = max(abs(aperture{m}(:)));
            probe_max_sub{m} = max(abs(sub_ap{m}(:)));
%% Create new exitwave
            buffer_exit_wave{m} = rspace.*aperture{m};
            buffer_exit_wave_sub{m} = rspace_sub .* sub_ap{m};
            temp_dp{m} = fft2(buffer_exit_wave{m});
            temp_dp_sub{m} = fft2(buffer_exit_wave_sub{m});
        end
%% calculated magnitudes at scan position aper
        collected_mag = zeros([size(diffpats,1) size(diffpats,2)],cdp);
        for ii = 1:length(lambda)
        collected_mag(goodInds) = collected_mag(goodInds) + abs(temp_dp{ii}(goodInds)).^2 + abs(temp_dp_sub{ii}(goodInds)).^2;
        end
%% re-weight the magnitudes
        mag_ratio = sqrt(complex(current_dp(goodInds))) ./ sqrt(complex(collected_mag(goodInds)));
        for m = 1:length(lambda)
            temp_dp{m}(goodInds) = mag_ratio .* temp_dp{m}(goodInds);
            temp_dp_sub{m}(goodInds) = mag_ratio .* temp_dp_sub{m}(goodInds);
%% Update the object
            new_exit_wave = ifft2(temp_dp{m});
            new_exit_wave_sub = ifft2(temp_dp_sub{m});
            diff_exit_wave = new_exit_wave - buffer_exit_wave{m};
            diff_exit_wave_sub = new_exit_wave_sub - buffer_exit_wave_sub{m};
            update_factor_ob{m} = conj(aperture{m}) ./ (probe_max{m}.^2);
            update_factor_ob_sub{m} = conj(sub_ap{m}) ./ (probe_max_sub{m}.^2);
            new_rspace = buffer_rspace{m} + update_factor_ob{m}.*beta_obj.*diff_exit_wave;
            new_rspace_sub = buffer_rspace_sub{m} + update_factor_ob_sub{m} .* beta_obj .* diff_exit_wave_sub;
            big_obj{m}(cropR(aper,:,m), cropC(aper,:,m)) = new_rspace;
            sub_obj{m}(cropR_sub(aper,:,m), cropC_sub(aper,:,m)) = new_rspace_sub;

%% Update the probe
        
            if itt > update_aperture_after && updateAp == 1
                update_factor_pr = beta_ap ./ object_max{m}.^2;
                update_factor_pr_sub = beta_ap ./ object_max_sub{m}.^2;
                if probe_refinement_flag == 1
                    ap_updated = aperture{m} +update_factor_pr*conj(buffer_rspace{m}).*(diff_exit_wave);
                    if pixel_size(central_mode) > pixel_size(m) %higher energy than central mode
                        central_probe = ifftn(fftn(aperture{central_mode}).*H_bk_shifted);
                        central_probe = central_probe(scoop_vec{m},scoop_vec{m});
                        Fprobe = my_fft(aperture{m}) .* H_bk{m};
                        Fprobe(scoop_vec{m},scoop_vec{m}) = my_fft(central_probe);
                        probe_rpl = my_ifft(Fprobe);
                        probe_rpl = ifftn(fftn(probe_rpl).*H_fwd{m});
                    elseif pixel_size(central_mode) < pixel_size(m) %lower energy than central mode
                        Fcentral_probe = my_fft(aperture{central_mode}).*H_bk{central_mode};
                        Fcentral_probe_cropped = Fcentral_probe(scoop_vec{m}, scoop_vec{m});
                        probe_rpl = zeros(little_area,cdp); %match class of other arrays
                        probe_rpl(scoop_vec{m},scoop_vec{m}) = my_ifft(Fcentral_probe_cropped);
    %                     probe_rpl = my_ifft(my_fft(probe_rpl).*H_fwd{m});
                        probe_rpl = ifftn(fftn(probe_rpl).*H_fwd{m});
                    else
                        probe_rpl = ap_updated;
                    end
                    ap_buffer = ap_updated + prb_rplmnt_weight(itt)*(probe_rpl-ap_updated);
                    aperture{m} = norm(ap_updated,'fro')/norm(ap_buffer,'fro')...
                        .*ap_buffer;
                else
                    aperture{m} = aperture{m} +update_factor_pr*conj(buffer_rspace{m}).*diff_exit_wave;
                end
                sub_ap{m} = sub_ap{m} + update_factor_pr_sub .* conj(buffer_rspace_sub{m}).*diff_exit_wave_sub;
            end 

 
  %% update the weights
            S(m) = sum(abs(aperture{m}(:)).^2);  
        end
        fourier_error(itt,aper) = sum(abs(sqrt(complex(current_dp(goodInds)))...
            - sqrt(complex(collected_mag(goodInds)))))./sum(sqrt(complex(current_dp(goodInds))));
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
            for m = 1:nModes
            best_obj{m} = big_obj{m};
            end
            best_err = mean_err;
        end

        if save_intermediate == 1 && mod(itt,floor(iterations/10)) == 0
            big_obj_g = cellfun(@gather, big_obj, 'UniformOutput', false);
            aperture_g = cellfun(@gather, aperture, 'UniformOutput', false);
            save([save_string filename '_itt' num2str(itt) '.mat'],...
                'big_obj_g','aperture_g','-v7.3');
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
    save([save_string filename '.mat'],...
        'best_obj','aperture','big_obj','initial_aperture','fourier_error','S','inter_obj','-v7.3');
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

%% Fresnel propogation
%     function U = fresnel_advance (U0, dx, dy, z, lambda)
%         % The function receives a field U0 at wavelength lambda
%         % and returns the field U after distance z, using the Fresnel
%         % approximation. dx, dy, are spatial resolution.
%         
%         k=2*pi/lambda;
%         [ny, nx] = size(U0);
%         
%         Lx = dx * nx;
%         Ly = dy * ny;
%         
%         dfx = 1./Lx;
%         dfy = 1./Ly;
%         
%         u = ones(nx,1)*((1:nx)-nx/2)*dfx;
%         v = ((1:ny)-ny/2)'*ones(1,ny)*dfy;
%         
%         O = my_fft(U0);
%         
%         H = exp(1i*k*z).*exp(-1i*pi*lambda*z*(u.^2+v.^2));
%         
%         U = my_ifft(O.*H);
%     end

%% Make a circle of defined radius

    function out = makeCircleMask(radius,imgSize)
        
        
        nc = imgSize/2+1;
        n2 = nc-1;
        [xx, yy] = meshgrid(-n2:n2-1,-n2:n2-1);
        R = sqrt(xx.^2 + yy.^2);
        out = R<=radius;
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



