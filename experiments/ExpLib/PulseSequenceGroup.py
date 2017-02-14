__author__ = 'Nitrogen'

def cphase(psb,control_id ,minus_Z_id):
    psb.append('q,mm'+str(control_id),'pi_ge')
    psb.append('q,mm'+str(minus_Z_id),'2pi_ef')
    psb.append('q,mm'+str(control_id),'pi_ge')


def cphase_v1(psb,control_id ,minus_Z_id,cz_phase=0):
    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge')
    psb.append('q,mm'+str(minus_Z_id),'pi_ef')
    psb.append('q,mm'+str(minus_Z_id),'pi_ef', phase=cz_phase)

    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge')


def cphase_v2(psb,control_id ,minus_Z_id,cz_phase1=0,cz_phase2=0):
    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge')
    psb.append('q,mm'+str(minus_Z_id),'pi_ef')
    psb.append('q,mm'+str(minus_Z_id),'pi_ef', phase=cz_phase1)

    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge')


#### Final CZ gate
def cphase_v3(psb,control_id ,minus_Z_id, efsbphase_0=0,efsbphase_1=0,gesbphase1=0,add_freq=0,efsbphase_2=0):
    # pass
    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge', phase = -gesbphase1,add_freq=add_freq)
    psb.append('q,mm'+str(minus_Z_id),'pi_ef',phase= efsbphase_0 )
    psb.append('q,mm'+str(minus_Z_id),'pi_ef',phase = efsbphase_2)
    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge',phase=efsbphase_1 + gesbphase1,add_freq=add_freq)



def cnot_v1(psb,control_id ,minus_Z_id, cnot_phase=0,cnot_dc_phase=0):
    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge')
    psb.append('q,mm'+str(minus_Z_id),'pi_ef')
    psb.append('q','pi_q_ef', phase=cnot_phase)
    psb.append('q,mm'+str(minus_Z_id),'pi_ef')

    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge',phase=cnot_dc_phase)


#### Final CNOT gate
def cnot_v2(psb,control_id ,minus_Z_id, cnot_phase=0,efsbphase_0=0,efsbphase_1=0,gesbphase1=0,efsbphase_2=0):
    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge', phase = -gesbphase1)
    psb.append('q,mm'+str(minus_Z_id),'pi_ef',phase= efsbphase_0 )
    psb.append('q','pi_q_ef', phase=cnot_phase)
    psb.append('q,mm'+str(minus_Z_id),'pi_ef',phase=efsbphase_2)

    if not control_id == "q":
        psb.append('q,mm'+str(control_id),'pi_ge',phase=efsbphase_1 + gesbphase1)

