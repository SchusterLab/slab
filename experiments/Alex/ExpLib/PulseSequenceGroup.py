__author__ = 'Nitrogen'

def cphase(psb,control_id ,minus_Z_id):
    psb.append('q,mm'+str(control_id),'pi_ge')
    psb.append('q,mm'+str(minus_Z_id),'2pi_ef')
    psb.append('q,mm'+str(control_id),'pi_ge')



# def cnot(psb,control_id ,minus_Z_id):
#     psb.append('q,mm'+str(control_id),'pi_ge')
#     psb.append('q,mm'+str(minus_Z_id),'pi_ef')
#     # self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq)
#     psb.append('q,mm'+str(control_id),'pi_ef',phase=180)
#     psb.append('q,mm'+str(control_id),'pi_ge',phase=180)