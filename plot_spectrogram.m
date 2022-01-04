% plot_spectrogram function is aimed to observe at the spectrogram of the data
% and use it too to identify informative frequency bands
% Inputs:
    % left_data = signal of the left hand imagination
    % right_data = signal of the right hand imagination
        %%%Note that the mark over the to latter varribales does not mean
        %%%they were not used. they are called as a dynamic input to create_spectrogram_vars functions.%%%  
    % electrodes = cell of names of the electrodes.
    % freq = frequncies vector.
    % window = window size.
    % overlap = overlap size.
    % Fs = Sampling rate.
    % classes = vector of data classes (left&right)
% Outputs:
    % mean_power_dif = the mean of the signal power difference between
        % electrodes C3 and C4.
    % two figures: one of the spectrogram plots for both hands and both electrode,
    % and the other of thespectrogram of the power difference.

function mean_power_dif = plot_spectrogram(left_data, right_data, electrodes,...
                                            freq, window, overlap, Fs, classes)
    % create windows data
    left_data = permute(left_data, [1 3 2]);
    right_data = permute(right_data, [1 3 2]);
    all_rec_data = cat(3, left_data, right_data)
    num_windows = floor((size(right_data,2) - window)/(window-overlap))+1; % Number of windows.

    % Plot design details
    plot_pos =  [1, 1, 16, 16];
    diff_plot_pos =  [1, 2, 16, 7.5];
    cb_pos = [0.92 ,0.10, 0.01, 0.8];
    Fontsize = 15;
    cb_Fontsize = 13;
    title_Fontsize = 20;
                                        
    % define hands and electrodes data.
    elec_num = length(electrodes);
    num_classes = length(classes);
    
    % Allocate data structure.
    mean_power = cell(elec_num,num_classes);
    mean_power_dif = cell(elec_num,1);
    
    % plot the power spectum (4 plots)
    figure('name', 'Power Spectrogram', 'NumberTitle', 'off', 'units', 'centimeters', 'Position', plot_pos)
    sgtitle('Power Spectrogram', 'FontSize', title_Fontsize);
    for elec = 1:elec_num
        for hand = 1:num_classes
            plot_ind = (elec-1)*2 + hand;
            subplot(elec_num, num_classes, plot_ind);
            [t, f, mean_power{elec, hand}] = create_spectrogram_vars(all_rec_data(:,:,2*hand-1:hand*2), elec, freq, num_windows, window, overlap, Fs);
            imagesc(t, f, mean_power{elec, hand}); % present data (PLOT)
            set(gca, 'YDir','normal')
            
            % coloring
            colormap(jet) 
            colorbar('visible' , 'off');
            
            % plot properties
            hold on
            title([lower(classes{hand}) ' hand ' electrodes{elec}], 'FontSize', Fontsize);
            
            if mod(plot_ind, num_classes) == 1
                ylabel('Frequency [Hz]', 'FontSize', Fontsize);
            end
            if elec == elec_num
                xlabel('Time [Sec]' ,'FontSize', Fontsize)
            end
        end 
    end
    cb_1 = colorbar('location' , 'manual', 'Position' , cb_pos);
    cb_1.Label.String = 'Power [dB]';
    cb_1.Label.FontSize = cb_Fontsize;
    
    % plot the power spectum difference (2 plots)
    figure('name', 'Power Spectrogram difference (Right-Left)', 'NumberTitle', 'off', 'units', 'centimeters','Position', diff_plot_pos);
    sgtitle('Power Spectrogram difference (Right-Left)');
    for elec = 1:elec_num
        subplot(1, elec_num, elec);
        mean_power_dif{elec} = abs(mean_power{elec, 1} - mean_power{elec, 2}); % calculate power difference. (right minus left)
        imagesc(t, f, mean_power_dif{elec});  % present data (PLOT)
        set(gca, 'YDir','normal')
        
        % coloring
        colormap(jet) 
        colorbar('visible' , 'off')
        hold on 
        
        % plot properties
        title([electrodes{elec}, ' Difference'], 'FontSize', Fontsize);
        xlabel('Time [Sec]' ,'FontSize', Fontsize);
        ylabel('Frequency [Hz]', 'FontSize', Fontsize);    
    end
    
     cb_2 = colorbar('location' , 'manual', 'Position' , cb_pos);
     cb_2.Label.String = 'Power difference [dB]';
     cb_2.Label.FontSize = cb_Fontsize;
end 