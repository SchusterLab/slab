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

def gauss_phase_fix(wtpts,mtpts,origin,marker_start_buffer,marker_end_buffer,pulse_location,pulse,pulse_info):

    if pulse_info['fix_phase']:
        print "Phase of qubit pulse is being fixed"
        qubit_dc_offset = pulse_info['qubit_dc_offset']
        print "qubit DC offset is %s" %(qubit_dc_offset)
        qubit_waveforms = ap.sideband(wtpts,
                                 ap.gauss(wtpts, pulse.amp,
                                          origin - pulse_location - 0.5*pulse.span_length,
                                          pulse.length), np.zeros(len(wtpts)),
                                 pulse.freq, phase= 360*qubit_dc_offset/1e9*(wtpts+pulse_location)+ pulse.phase)
        qubit_marker = ap.square(mtpts, 1,
                                                  origin - pulse_location - pulse.span_length - marker_start_buffer,
                                                  pulse.span_length + marker_start_buffer- marker_end_buffer)
        return (qubit_waveforms,qubit_marker)




    else:
        qubit_waveforms = ap.sideband(wtpts,
                                 ap.gauss(wtpts, pulse.amp,
                                          origin - pulse_location - 0.5*pulse.span_length,
                                          pulse.length), np.zeros(len(wtpts)),
                                 pulse.freq, pulse.phase)
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

    if flux_pulse_info['fix_phase']:

        if pulse.name[-1] == "f":
            dc_offset = mm_target_info['dc_offset_freq_ef']
            shifted_frequency = mm_target_info['flux_pulse_freq_ef']
            bare_frequency = shifted_frequency-dc_offset
            waveforms_qubit_flux = ap.sideband(ftpts,
                                             ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, pulse_cfg['square'][
                                                                          'ramp_sigma']), np.zeros(len(ftpts)),freq=bare_frequency, phase= 360*dc_offset/1e9*(ftpts+pulse_location)+pulse.phase,offset=False)[0]

        else:
            dc_offset = mm_target_info['dc_offset_freq']
            shifted_frequency = mm_target_info['flux_pulse_freq']
            bare_frequency = shifted_frequency-dc_offset
            waveforms_qubit_flux = ap.sideband(ftpts,
                                             ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, pulse_cfg['square'][
                                                                          'ramp_sigma']), np.zeros(len(ftpts)),freq=bare_frequency, phase= 360*dc_offset/1e9*(ftpts+pulse_location)+pulse.phase,offset=False)[0]



    else:
        waveforms_qubit_flux = ap.sideband(ftpts,
                                           ap.square(ftpts, pulse.amp, pulse_location-pulse.length-0.5*(pulse.span_length - pulse.length) , pulse.length, pulse_cfg['square'][
                                                                      'ramp_sigma']), np.zeros(len(ftpts)),
                                          pulse.freq, pulse.phase,offset=False)[0]


    return waveforms_qubit_flux



def flux_gauss(ftpts,pulse_location,pulse):
    waveforms_qubit_flux = ap.sideband(ftpts,
                             ap.gauss(ftpts, pulse.amp,
                                      pulse_location - 0.5*pulse.span_length,
                                      pulse.length), np.zeros(len(ftpts)),
                             pulse.freq, pulse.phase)[1]
    return waveforms_qubit_flux