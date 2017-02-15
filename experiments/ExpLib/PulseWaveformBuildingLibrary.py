from slab.experiments.ExpLib import awgpulses as ap
import numpy as np

def square(wtpts,mtpts,origin,marker_start_buffer,marker_end_buffer,pulse_location,pulse,pulse_cfg):
    qubit_waveforms = ap.sideband(wtpts,
                             ap.square(wtpts, pulse.amp,
                                       origin - pulse_location - pulse.length - 0.5*(pulse.span_length - pulse.length), pulse.length,
                                       pulse_cfg['square']['ramp_sigma']),
                             np.zeros(len(wtpts)),
                             pulse.freq, pulse.phase)
    qubit_marker = ap.square(mtpts, 1, origin - pulse_location - pulse.span_length - marker_start_buffer,
                                                              pulse.span_length + marker_start_buffer - marker_end_buffer)
    return (qubit_waveforms,qubit_marker)


def gauss(wtpts,mtpts,origin,marker_start_buffer,marker_end_buffer,pulse_location,pulse):
    qubit_waveforms = ap.sideband(wtpts,
                             ap.gauss(wtpts, pulse.amp,
                                      origin - pulse_location - 0.5*pulse.span_length,
                                      pulse.length), np.zeros(len(wtpts)),
                             pulse.freq, pulse.phase)
    qubit_marker = ap.square(mtpts, 1,
                                              origin - pulse_location - pulse.span_length - marker_start_buffer,
                                              pulse.span_length + marker_start_buffer- marker_end_buffer)
    return (qubit_waveforms,qubit_marker)

def gauss_phase_fix(wtpts,mtpts,origin,marker_start_buffer,marker_end_buffer,pulse_location,pulse,pulse_info,qubit_dc_offset,t0=0):

    if pulse_info['fix_phase']:
        # print "Phase of qubit pulse is being fixed"
        # print "qubit DC offset is %s" %(qubit_dc_offset)
        qubit_waveforms = ap.sideband(wtpts,
                                 ap.gauss(wtpts, pulse.amp,
                                          origin - pulse_location - 0.5*pulse.span_length,
                                          pulse.length), np.zeros(len(wtpts)),
                                 pulse.freq, phase= 360*(qubit_dc_offset+pulse.add_freq)/1e9*(wtpts+pulse_location)+ pulse.phase)
        qubit_marker = ap.square(mtpts, 1,
                                                  origin - pulse_location - pulse.span_length - marker_start_buffer,
                                                  pulse.span_length + marker_start_buffer- marker_end_buffer)
        return (qubit_waveforms,qubit_marker)




    else:
        qubit_waveforms = ap.sideband(wtpts,
                                 ap.gauss(wtpts, pulse.amp,
                                          origin - pulse_location - 0.5*pulse.span_length,
                                          pulse.length), np.zeros(len(wtpts)),
                                 pulse.freq, pulse.phase,t0=t0)
        qubit_marker = ap.square(mtpts, 1,
                                                  origin - pulse_location - pulse.span_length - marker_start_buffer,
                                                  pulse.span_length + marker_start_buffer- marker_end_buffer)
        return (qubit_waveforms,qubit_marker)

def flux_square(ftpts,pulse_location,pulse,pulse_cfg):

    waveforms_qubit_flux = ap.sideband(ftpts,
                                     ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, pulse_cfg['square'][
                                                                  'ramp_sigma']), np.zeros(len(ftpts)),
                                      pulse.freq, pulse.phase,offset=False)[0]
    return waveforms_qubit_flux

def flux_square_phase_fix(ftpts,pulse_location,pulse,pulse_cfg, mm_target_info, flux_pulse_info):

    if flux_pulse_info['chirp'] and flux_pulse_info['fix_phase']:
        if pulse.name[-1] == "f":
            dc_offset = mm_target_info['dc_offset_freq_ef']
            shifted_frequency = mm_target_info['flux_pulse_freq_ef']
            offset_fit_quad_ef=dc_offset/(mm_target_info['a_ef']**2)
            time_step = ftpts[1]-ftpts[0]
            f_integ_array = 360.0*np.cumsum(offset_fit_quad_ef*ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, mm_target_info['ramp_sigma_ef'])**2)*time_step/1.0e9
            # print f_integ_array[-1]
            # print f_integ_array[0]
            # print np.shape(f_integ_array)

            bare_frequency = shifted_frequency+dc_offset
            waveforms_qubit_flux = ap.sideband(ftpts,
                                             ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, mm_target_info['ramp_sigma_ef']), np.zeros(len(ftpts)),freq=bare_frequency, phase= 360*(pulse.add_freq)/1e9*(ftpts-pulse_location)+pulse.phase - f_integ_array ,offset=True,offset_fit_quad=offset_fit_quad_ef)[0]

        else:
            dc_offset = mm_target_info['dc_offset_freq']
            print dc_offset
            shifted_frequency = mm_target_info['flux_pulse_freq']
            bare_frequency = shifted_frequency+dc_offset
            waveforms_qubit_flux = ap.sideband(ftpts,
                                             ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length,mm_target_info['ramp_sigma']), np.zeros(len(ftpts)),freq=bare_frequency, phase= 360*(-dc_offset+pulse.add_freq)/1e9*(ftpts-pulse_location)+pulse.phase,offset=False)[0]


    elif flux_pulse_info['fix_phase']:

        if pulse.name[-1] == "f":
            dc_offset = mm_target_info['dc_offset_freq_ef']
            shifted_frequency = mm_target_info['flux_pulse_freq_ef']
            bare_frequency = shifted_frequency+dc_offset
            # print mm_target_info['ramp_sigma_ef']
            waveforms_qubit_flux = ap.sideband(ftpts,
                                             ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, mm_target_info['ramp_sigma_ef']), np.zeros(len(ftpts)),freq=bare_frequency, phase= 360*(-dc_offset + pulse.add_freq)/1e9*(ftpts-pulse_location)+pulse.phase,offset=False)[0]

        else:
            dc_offset = mm_target_info['dc_offset_freq']
            shifted_frequency = mm_target_info['flux_pulse_freq']
            bare_frequency = shifted_frequency+dc_offset
            waveforms_qubit_flux = ap.sideband(ftpts,
                                             ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length,mm_target_info['ramp_sigma']), np.zeros(len(ftpts)),freq=bare_frequency, phase= 360*(-dc_offset+pulse.add_freq)/1e9*(ftpts-pulse_location)+pulse.phase,offset=False)[0]

    else:
        if pulse.name[-1] == "f":
            waveforms_qubit_flux = ap.sideband(ftpts,
                                               ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, mm_target_info['ramp_sigma_ef']), np.zeros(len(ftpts)),
                                              mm_target_info['flux_pulse_freq_ef'], pulse.phase,offset=False)[0]

        else:
            waveforms_qubit_flux = ap.sideband(ftpts,
                                   ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, mm_target_info['ramp_sigma']), np.zeros(len(ftpts)),
                                  pulse.freq, pulse.phase,offset=False)[0]

    # print np.shape(waveforms_qubit_flux)

    return waveforms_qubit_flux



def flux_gauss(ftpts,pulse_location,pulse):
    waveforms_qubit_flux = ap.sideband(ftpts,
                             ap.gauss(ftpts, pulse.amp,
                                      pulse_location - 0.5*pulse.span_length,
                                      pulse.length), np.zeros(len(ftpts)),
                             pulse.freq, pulse.phase)[1]
    return waveforms_qubit_flux