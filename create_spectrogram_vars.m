% create_spectrogram_vars performs the spectogram creation and returns the
% relevent varriable nedded for futher calculation and manipulation, and
% plots.
% Inputs:
    % data = the signal from the relevent hand
    % elec_num = number of electrodes.
    % freq = frequncies vector.
    % num_windows = number of windows to split the data to calcuate signal
        % data 
    % window = window size.
    % overlap = overlap size.
    % Fs = Sampling rate.
% Outputs:
    % t =  a vector of times at which the spectrogram function computes
    % f =  a vector of frequencies at which the spectrogram function computes
    % mean_power  = the mean of the signal power in a certain electrode.
    
function [t, f, mean_power] = create_spectrogram_vars(data, elec_num, freq, num_windows, window, overlap, Fs)
    num_windows = 345
    power = zeros(size(data, 1), length(freq), num_windows); % alocate matrix to insert data

     % going over samples data.
     for i = 1:size(data,1) 
        [s, f, t] = spectrogram(data(i,:,elec_num), window, overlap, freq, Fs, 'power', 'yaxis'); % Calculate spectogram
        power(i,:,:) = s .* conj(s); % Calculate power.
     end
    
    power = 10*log10(power);
    mean_pow = mean(power, 1);
    mean_power = squeeze(mean_pow);
end
