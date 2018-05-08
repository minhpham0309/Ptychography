%difference map
function [big_obj,aperture,fourier_error,initial_obj,initial_aperture,inter_obj] = DM1(ePIE_inputs,varargin)
%varargin = {beta_obj, beta_ap}
optional_args = {0.9 0.1 0}; %default values for varargin parameters
rng('shuffle','twister');
%% setup working and save directories

dir = pwd;
save_string = [ dir '/Results_ptychography/']; % Place to save results
imout = 1; % boolean to determine whether to monitor progress or not
% [~,jobID] = system('echo $JOB_ID');
% jobID = jobID(~isspace(jobID));
%% Load inputs from struct
diffpats = ePIE_inputs(1).Patterns;
positions = ePIE_inputs(1).Positions;
pixel_size = ePIE_inputs(1).PixelSize;
big_obj = ePIE_inputs(1).InitialObj;
aperture_radius = ePIE_inputs(1).ApRadius;
aperture = ePIE_inputs(1).InitialAp;
iterations = ePIE_inputs(1).Iterations;
%% parameter inputs
if isfield(ePIE_inputs, 'updateAp'), update_aperture = ePIE_inputs.updateAp;
else update_aperture = 1;
end

if isfield(ePIE_inputs, 'GpuFlag'), gpu = ePIE_inputs(1).GpuFlag;
else gpu = 0;
end

if isfield(ePIE_inputs, 'miscNotes'), miscNotes = ePIE_inputs.miscNotes;
else miscNotes = 'None';
end

if isfield(ePIE_inputs, 'showim'), showim = ePIE_inputs(1).showim;
else showim = 0;
end

if isfield(ePIE_inputs, 'do_posi');do_posi = ePIE_inputs.do_posi;
else do_posi = 0;
end

if isfield(ePIE_inputs, 'save_intermediate'); save_intermediate = ePIE_inputs.save_intermediate;
else save_intermediate = 0;
end

if isfield(ePIE_inputs, 'update_probe_after'); update_probe_after = ePIE_inputs.update_probe_after;
else update_probe_after = Inf;
end

clear ePIE_inputs

%% === Reconstruction parameters frequently changed === %%
nva = length(varargin);
optional_args(1:nva) = varargin;
[beta_obj, beta_ap, probe_norm] = optional_args{:};
%% print parameters
fprintf('iterations = %d\n', iterations);
fprintf('beta probe = %0.1f\n', beta_ap);
fprintf('beta obj = %0.1f\n', beta_obj);
fprintf('gpu flag = %d\n', gpu);
fprintf('updating probe = %d\n', update_aperture);
fprintf('positivity = %d\n', do_posi);
fprintf('misc notes: %s\n', miscNotes);
%% Define parameters from data and for reconstruction
for ii = 1:size(diffpats,3)
    diffpats(:,:,ii) = ifftshift(sqrt(diffpats(:,:,ii)));
    %diffpats(:,:,ii) = sqrt(diffpats(:,:,ii));
end
diffpats = single(diffpats);
[y_kspace,~] = size(diffpats(:,:,1)); % Size of diffraction patterns
nApert = size(diffpats,3);

little_area = y_kspace; % Region of reconstruction for placing back into full size image
%% Get centre positions for cropping (should be a 2 by n vector)
%positions = convert_to_pixel_positions(positions,pixel_size);
[pixelPositions, bigx, bigy] = ...
    convert_to_pixel_positions_testing5(positions,pixel_size,little_area);
centrey = round(pixelPositions(:,2));
centrex = round(pixelPositions(:,1));
centBig = round((bigx+1)/2);
Y1 = centrey - floor(y_kspace/2); Y2 = Y1+y_kspace-1;
X1 = centrex - floor(y_kspace/2); X2 = X1+y_kspace-1;
%% create initial aperture and object guesses
if aperture == 0
    aperture = single(((makeCircleMask(round(aperture_radius./pixel_size),little_area))));
else
    aperture = single(aperture);
end
initial_aperture = aperture;

if big_obj == 0
    big_obj = single(rand(bigx,bigy)).*exp(1i*(rand(bigx,bigy)));
else
    big_obj = single(big_obj);
end
initial_obj = big_obj;

if save_intermediate == 1
    inter_obj = zeros([size(big_obj) floor(iterations/10)]);
    inter_frame = 0;
else
    inter_obj = [];
end

fourier_error = zeros(iterations,nApert);

if gpu == 1
    display('========DM reconstructing with GPU========')
    diffpats = gpuArray(diffpats);
    fourier_error = gpuArray(fourier_error);
    big_obj = gpuArray(big_obj);
    aperture = gpuArray(aperture);
else
    display('========DM reconstructing with CPU========')
end

best_err = 100; % check to make sure saving reconstruction with best error

% make a big_obj counter
big_obj_counter = zeros(size(big_obj));
for ii = 1:nApert
    r = Y1(ii):Y2(ii);
    c = X1(ii):X2(ii);
    big_obj_counter(r,c) = big_obj_counter(r,c) + 1;
end
big_obj_counter(big_obj_counter==0)=1;


%% make state vector
psi = zeros(y_kspace,y_kspace,nApert);
for ii = 1:nApert
    psi(:,:,ii) = aperture .* big_obj(Y1(ii):Y2(ii),X1(ii):X2(ii));
end

%% Main ePIE itteration loop
disp('========beginning reconstruction=======');
for itt = 1:iterations

    big_obj_numer = zeros(size(big_obj));
    big_obj_denom= zeros(size(big_obj));
    probe_numer = zeros(y_kspace);
    probe_denom= zeros(y_kspace);   
    abs_aperture_2 =  abs(aperture).^2;
    
    % update object
    for ii=1:nApert
        r = Y1(ii):Y2(ii);
        c = X1(ii):X2(ii);
        u_old = big_obj(r, c);
        psi_old = psi(:,:,ii);
        current_dp = diffpats(:,:,ii);        
        
        psi_O = aperture .* u_old;
        check_dp = abs(fft2(psi_O));
        missing_data = current_dp == -1;
        fourier_error(itt,ii) = sum(sum(abs(current_dp(~missing_data) - check_dp(~missing_data))))./sum(sum(current_dp(~missing_data)));
               
        z = fft2( 2*psi_O - psi_old );
        z_missing = z(missing_data);
        z = diffpats(:,:,ii) .* exp(1i*angle(z));
        z(missing_data)=z_missing;
        psi_F = ifft2(z);
        psi(:,:,ii) = psi_old + (psi_F - psi_O);

        %check_dp = abs(fft2(psi(:,:,ii)));
        %fourier_error(itt,ii) = sum(sum(abs(current_dp(~missing_data) - check_dp(~missing_data))))./sum(sum(current_dp(~missing_data)));

        % update object_i
        %u_new = ( (1-beta_obj)*u_old + dt*psi(:,:,ii).*conj(aperture) ) ./ ( (1-beta_obj) + dt*abs(aperture).^2  );
        %big_obj(r,c)  = u_new;
        big_obj_numer(r,c) = big_obj_numer(r,c) + psi(:,:,ii).*conj(aperture);
        big_obj_denom(r,c)= big_obj_denom(r,c) + abs_aperture_2;
        
        % update probe
        %new_beta_ap = beta_ap*sqrt((iterations-itt)/iterations);        
        %ds = new_beta_ap./object_max;
        %probe_temp = probe_temp + conj(u_old).*psi(:,:,ii);
        %probe_denom= probe_denom+ abs(u_old).^2;
        %aperture = aperture - ds*conj(u_old).*(psi_old - psi(:,:,ii));
        %aperture = ((1-new_beta_ap)*aperture + ds*psi(:,:,ii).*conj(u_old)) ./ ( (1-new_beta_ap) + ds*abs(u_old).^2 );      
        %big_obj_temp(r,c) = big_obj_temp(r,c) + u_new;
    end
    probe_max = max(abs(aperture(:))).^2;
    %big_obj = big_obj_numer./(big_obj_denom+1e-3);
    big_obj = ((1-beta_obj)*big_obj + (beta_obj/probe_max/nApert)* big_obj_numer) ./ (1-beta_obj + (beta_obj/probe_max/nApert)*big_obj_denom);
    
    % update probe
    for ii=1:nApert
        r = Y1(ii):Y2(ii);
        c = X1(ii):X2(ii);
        u_old = big_obj(r, c);
        %new_beta_ap = beta_ap*sqrt((iterations-itt)/iterations);        
        %ds = new_beta_ap./object_max;
        probe_numer = probe_numer + conj(u_old).*psi(:,:,ii);
        probe_denom= probe_denom+ abs(u_old).^2; 
    end
    object_max = max(abs(big_obj(:))).^2;
    %aperture=probe_temp./probe_denom;
    aperture = ((1-beta_ap)*aperture + (beta_ap/object_max/nApert)*probe_numer )./ (1-beta_ap + (beta_ap/object_max/nApert)* probe_denom);
    %new_beta_ap = beta_ap*sqrt((iterations-itt)/iterations); 
    %aperture = ((1-new_beta_ap)*aperture + (new_beta_ap/object_max/nApert)* probe_numer) ./ (1-new_beta_ap + (new_beta_ap/object_max/nApert)*probe_denom);
    
    if probe_norm
        scale = max(abs(aperture(:))); aperture = aperture/scale;
    end
    
    figure(99);imagesc(abs(big_obj)),axis image; colormap gray
    
    if  mod(itt,showim) == 0 && imout == 1;      
        figure(3)        
        hsv_big_obj = make_hsv(big_obj,1);
        hsv_aper = make_hsv(aperture,1);
        subplot(2,2,1)
        imagesc(abs(big_obj)); axis image; colormap gray; title(['reconstruction pixel size = ' num2str(pixel_size)] )
        subplot(2,2,2)
        imagesc(hsv_aper); axis image; colormap gray; title('aperture single'); colorbar
        subplot(2,2,3)
        errors = sum(fourier_error,2)/nApert;
        fprintf('%d. Error = %f, scale = %f\n',itt,errors(itt),max(max(abs(aperture))));
        plot(errors); ylim([0,0.2]);
        subplot(2,2,4)
        imagesc(log(fftshift(check_dp))); axis image
        drawnow
        
        
    end
    mean_err = sum(fourier_error(itt,:),2)/nApert;
    if best_err > mean_err
        best_obj = big_obj;
        best_err = mean_err;
    end
    if save_intermediate == 1 && mod(itt,10) == 0
        inter_frame = inter_frame + 1;
        inter_obj(:,:,inter_frame) = big_obj;
    end
end
disp('======reconstruction finished=======')



% if saveOutput == 1
%     save([save_string 'best_obj_' filename '.mat'],'best_obj','aperture','initial_obj','initial_aperture','fourier_error');
% end

%% Function for converting positions from experimental geometry to pixel geometry

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
        hue = (hue./max(hue(:)));
        value = (value./max(value(:))).*factor;
        hsv_obj(:,:,1) = hue;
        hsv_obj(:,:,3) = value;
        hsv_obj(:,:,2) = ones(sizey,sizex);
        hsv_obj = hsv2rgb(hsv_obj);
    end
%% Function for defining a specific region of an image

    function [roi] = get_roi(image, centrex,centrey,crop_size)
        
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

end


