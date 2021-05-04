import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.special

def cav_response_new(p, x):
    """(p[0]/p[1])/(-1/2*p[0]/p[1] - 1j*(x-p[0])"""
    ### p[0]=center freq, p[1]=kappa
    temp = (p[1])/(p[1] - 1j*(x-p[0]))
    return temp/max(abs(temp))

def IF_window(p, x):
    ### p[0] = center freq, p[1] = window width
    temp = np.zeros(len(x)) + 1j*np.zeros(len(x))
    for ii in range(len(x)):
        if x[ii]>(p[0]-p[1]) and x[ii]<(p[0]+p[1]):
            temp[ii] = 1/np.sqrt(2)*(1+1j)
        else:
            pass
    return temp/max(abs(temp))

def erf_t(A, sig, tc, tb, t):
    #A-Amplitude, sig-Gaussian Filter Width, tc-Core Pulse length, tb - zero-amplitude buffer length
    return (A/2)*(sc.special.erf((t-tb)/(np.sqrt(2)*sig))-sc.special.erf((t-tc-tb)/(np.sqrt(2)*sig)))

def optimal_rr(readout_params, flat_top_amp):
    """All the necessary readout parameters"""
    rr_f, rr_q, rr_len, dt, twochi = readout_params 

    kappa = rr_f/rr_q #linewidth
    cav_freq = 0 #generate the pulse in rotating frame
    n_points = int(rr_len/dt/5) #number of points
    pad_factor = 10
    t = dt * np.arange(0, n_points * pad_factor)
    ifreq = np.fft.fftfreq(n_points * pad_factor, dt)
    freq = np.fft.fftshift(ifreq)
    """Hard cut-off to constrain the BW of the output pulse to be within the AWG BW"""
    if_band = IF_window([cav_freq, 250e6], freq) 

    """Minimum pulse length required to resolve the qubit g and e peaks"""
    pulse_len = 1/twochi
    
    """Find the sigma of rising waveform for a given flat-top amplitude (they are somehow related)"""
    sigarray = np.arange(15e-9, 60e-9, 1.0e-9)
    ratio = []
    
    for sig in sigarray:
        desired_output = erf_t(1, sig, pulse_len, 1500e-9, t)
        
        desired_output_ifft = np.fft.fft(desired_output, n_points * pad_factor)/n_points
        desired_output_sfft = np.fft.fftshift(desired_output_ifft) #"sfft" denotes shifted spectrum to center at cav_freq

        lorenz_c = cav_response_new([cav_freq, kappa], freq)

        input_sfft = (desired_output_sfft/lorenz_c)*if_band

        output_sfft = input_sfft*lorenz_c
        output_fft = np.fft.ifftshift(output_sfft)
        output_pulse = np.fft.ifft(output_sfft)

        input_fft = np.fft.ifftshift(input_sfft)
        input_pulse = np.fft.ifft(input_fft)

        flip = input_pulse[::-1]

        opt_pulse = np.real(flip)

        max_opt = np.max(opt_pulse)
        mid_opt = opt_pulse[int(len(opt_pulse)/2)]
        ratio.append(mid_opt/max_opt)

    """Returns an array of the steady state amplitude as a function of sigma"""    
    ratio_targ = flat_top_amp
    ratio = np.array(ratio)
    index = np.argmin(abs(ratio-ratio_targ))

    """Returns the sigma"""
    desired_sigma = sigarray[index]

    """Now generate the desired output shape"""
    desired_output = erf_t(1, desired_sigma, pulse_len, 1500e-9, t)

    desired_output_ifft = np.fft.fft(desired_output,n_points*pad_factor)/n_points
    desired_output_sfft = np.fft.fftshift(desired_output_ifft) #"sfft" denotes shifted spectrum to center at cav_freq
    
    """Convolve with the Lorenztian transfer function of the cavity"""
    lorenz_c = cav_response_new([cav_freq, kappa], freq)
    
    input_sfft = (desired_output_sfft/lorenz_c)*if_band
 
    output_sfft = input_sfft * lorenz_c
    output_fft = np.fft.ifftshift(output_sfft)
    output_pulse = np.fft.ifft(output_sfft)

    input_fft = np.fft.ifftshift(input_sfft)
    input_pulse = np.fft.ifft(input_fft)
    flip = input_pulse[::-1]

    opt_pulse = np.real(flip)
    """Normalize the waveform"""
    opt_pulse = opt_pulse/np.max(opt_pulse)

    """Slice the waveform to remove padded zeros """
    result1 = np.where(opt_pulse > 0.00001)
    result2 = np.where(opt_pulse < -0.00001)

    #Trim Pulse
    start_trim = result1[0][0]
    end_trim = result2[0][len(result2[0])-1]
    numb = end_trim - start_trim  
    rem = numb % 4
    
    """Trim the pulse to be in multiples of 4ns, OPX AWG"""
    trim_pulse = opt_pulse[start_trim:end_trim - rem]
    check = len(trim_pulse) % 4
    if check !=0:
        print('Error: The final pulse is not a multiple of 4')
    
    return trim_pulse

"""Outputting an example pulse shape"""
read_params = [8.0517e9, 8400, 3.4e-6, 1e-9, 380e3]
s = optimal_rr(read_params, flat_top_amp=0.45)

plt.figure(dpi=300)
plt.plot(s)
plt.xlabel('Time (ns)')
plt.ylabel('AWG amp. (a.u.)')
plt.show()

print(len(s))