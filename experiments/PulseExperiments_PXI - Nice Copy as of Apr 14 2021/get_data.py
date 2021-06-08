import numpy as np
import scipy.optimize as spo
from sklearn import linear_model
from sklearn import metrics
from collections import Counter

from sklearn import linear_model
from sklearn import metrics
from collections import Counter
def get_singleshot_data_two_qubits_4_calibration_v2(single_data_list):


    X_list = []

    qubit_state = ['g','e']

    # print(single_data_list[0].shape)

    ii = -4

    for qubit_1_state in qubit_state:
        for qubit_2_state in qubit_state:

            qubit_1_ss_cos = single_data_list[0][0][0][ii]
            qubit_1_ss_sin = single_data_list[0][0][1][ii]
            qubit_2_ss_cos = single_data_list[1][1][0][ii]
            qubit_2_ss_sin = single_data_list[1][1][1][ii]

            X_state = np.array([qubit_1_ss_cos,qubit_1_ss_sin, qubit_2_ss_cos, qubit_2_ss_sin ])

            X_list.append(X_state)

            ii+=1

    Y_list = []

    for ii in range(4):
        Y_list.append(ii*np.ones(X_list[0].shape[1]))


    X = np.transpose(np.hstack(X_list))
    Y = np.vstack((Y_list)).flatten()


    ### copy from example
    h = .002  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5,multi_class='multinomial',solver='lbfgs')

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    Y_pred = logreg.predict(X)

    confusion_matrix = metrics.confusion_matrix(Y,Y_pred)

    # print(confusion_matrix)

    confusion_matrix_inv = np.linalg.inv(confusion_matrix)


    # print(single_data_list[0].shape[-2])
    counter_array_list = []
    for ii in range(single_data_list[0].shape[-2]-4):
        qubit_1_ss_cos = single_data_list[0][0][0][ii]
        qubit_1_ss_sin = single_data_list[0][0][1][ii]
        qubit_2_ss_cos = single_data_list[1][1][0][ii]
        qubit_2_ss_sin = single_data_list[1][1][1][ii]

        X_state = np.array([qubit_1_ss_cos,qubit_1_ss_sin, qubit_2_ss_cos, qubit_2_ss_sin ])

        data_X = np.transpose(X_state)
        data_Y = logreg.predict(data_X)

        counter = Counter(data_Y)
#         print(counter)

        counter_array = np.zeros(4)
        for key, value in counter.items():
            counter_array[int(key)] = value

        counter_array_list.append(counter_array)
#         print(counter_array)

    counter_array_list = np.array(counter_array_list)

#     print(counter_array_list[-9:,:])

#     print(np.dot(confusion_matrix_inv,counter_array_list[-9:,:]))

    # print(counter_array_list.shape)
    state_norm = np.dot(np.transpose(confusion_matrix_inv),np.transpose(counter_array_list))

#     print(state_norm[-9:,:])

    # plt.figure(figsize=(7,7))
    # plt.plot(state_norm[0],label='gg')
    # plt.plot(state_norm[1],label='ge')
    # plt.plot(state_norm[2],label='eg')
    # plt.plot(state_norm[3],label='ee')
    # plt.legend(bbox_to_anchor=(1, 0.8))

    return state_norm

def get_singleshot_data_two_qubits_9_calibration(single_data_list):


    X_list = []

    qubit_state = ['g','e','f']

    print(single_data_list[0].shape)

    ii = -9

    for qubit_1_state in qubit_state:
        for qubit_2_state in qubit_state:

            qubit_1_ss_cos = single_data_list[0][0][0][ii]
            qubit_1_ss_sin = single_data_list[0][0][1][ii]
            qubit_2_ss_cos = single_data_list[1][1][0][ii]
            qubit_2_ss_sin = single_data_list[1][1][1][ii]

            X_state = np.array([qubit_1_ss_cos,qubit_1_ss_sin, qubit_2_ss_cos, qubit_2_ss_sin ])

            X_list.append(X_state)

            ii+=1

    Y_list = []

    for ii in range(9):
        Y_list.append(ii*np.ones(X_list[0].shape[1]))


    X = np.transpose(np.hstack(X_list))
    Y = np.vstack((Y_list)).flatten()


    ### copy from example
    h = .002  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5,multi_class='multinomial',solver='lbfgs')

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    Y_pred = logreg.predict(X)

    confusion_matrix = metrics.confusion_matrix(Y,Y_pred)

    # print(confusion_matrix)

    confusion_matrix_inv = np.linalg.inv(confusion_matrix)


    # print(single_data_list[0].shape[-2])
    counter_array_list = []
    for ii in range(single_data_list[0].shape[-2]-9):
        qubit_1_ss_cos = single_data_list[0][0][0][ii]
        qubit_1_ss_sin = single_data_list[0][0][1][ii]
        qubit_2_ss_cos = single_data_list[1][1][0][ii]
        qubit_2_ss_sin = single_data_list[1][1][1][ii]

        X_state = np.array([qubit_1_ss_cos,qubit_1_ss_sin, qubit_2_ss_cos, qubit_2_ss_sin ])

        data_X = np.transpose(X_state)
        data_Y = logreg.predict(data_X)

        counter = Counter(data_Y)
#         print(counter)

        counter_array = np.zeros(9)
        for key, value in counter.items():
            counter_array[int(key)] = value

        counter_array_list.append(counter_array)
#         print(counter_array)

    counter_array_list = np.array(counter_array_list)

#     print(counter_array_list[-9:,:])

#     print(np.dot(confusion_matrix_inv,counter_array_list[-9:,:]))

    # print(counter_array_list.shape)
    state_norm = np.dot(np.transpose(confusion_matrix_inv),np.transpose(counter_array_list))

#     print(state_norm[-9:,:])

    # plt.figure(figsize=(7,7))
    # plt.plot(state_norm[0],label='gg')
    # plt.plot(state_norm[1],label='ge')
    # plt.plot(state_norm[2],label='gf')
    # plt.plot(state_norm[3],label='eg')
    # plt.plot(state_norm[4],label='ee')
    # plt.plot(state_norm[5],label='ef')
    # plt.plot(state_norm[6],label='fg')
    # plt.plot(state_norm[7],label='fe')
    # plt.plot(state_norm[8],label='ff')
    # plt.legend(bbox_to_anchor=(1, 0.8))

    return state_norm

def two_qubit_quantum_state_tomography(data):

    ind = 0#(state_index*operations_num*tomography_pulse_num)+(op_index*tomography_pulse_num)

    avgs = []

    avgs.append(1)
    for i in range(15):
        avgs.append(data[ind+i])

    #print "avgs2" + str(avgs2)

    #for testing
    #avgs=data
    #
    #print "avgs" + str(avgs)
    amp = np.sqrt(sum(np.square(avgs)))

    #print amp


    def get_P_array():
        ## Pauli Basis
        I = np.matrix([[1,0],[0,1]])
        X = np.matrix([[0,1],[1,0]])
        Y = np.matrix([[0,-1j],[1j,0]])
        Z = np.matrix([[1,0],[0,-1]])

        P=[]
        P.append(I)
        P.append(X)
        P.append(Y)
        P.append(Z)

        return P

    P = get_P_array()

    # tensor products of Pauli matrixes
    B=[]
    for i in range(4):
        for j in range(4):
            B.append(np.kron(P[i], P[j]))

    den_mat =(0.25*avgs[0]*B[0]).astype(np.complex128)
    for i in np.arange(1,16):
        den_mat += 0.25*avgs[i]*B[i]


    #
    # print("Density Matrix:")
    # print( den_mat)
    # print ("Trace:")
    # print (np.real(np.trace(den_mat)))
    # print ("Density Matrix Squared")
    # print( np.dot(den_mat,den_mat))
    # print( "Trace:")
    # print( np.real(np.trace(np.dot(den_mat,den_mat))))
    #
    #
    #     ### Generate 3D bar chart
    # ## real
    # fig = plt.figure(figsize=(20,20))
    #
    # ax = fig.add_subplot(111, title='Real', projection='3d')
    #
    # coord= [0,1,2,3]
    #
    # x_pos=[]
    # y_pos=[]
    # for i in range(4):
    #     for j in range(4):
    #         x_pos.append(coord[i])
    #         y_pos.append(coord[j])
    #
    # xpos=np.array(x_pos)
    # ypos=np.array(y_pos)
    # zpos=np.array([0]*16)
    # dx = [0.6]*16
    # dy = dx
    # dz=np.squeeze(np.asarray(np.array(np.real(den_mat).flatten())))
    #
    # nrm=mpl.colors.Normalize(-1,1)
    # colors=mpl.cm.Reds(nrm(dz))
    # alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)
    #
    # for i in range(len(dx)):
    #     ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
    # xticks=['ee','eg','ge','gg']
    # yticks=xticks
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0,1,2,3])
    # ax.set_yticklabels(yticks)
    # ax.set_zlim(-1,1)
    # plt.show()
    #
    # # imaginary
    #
    # fig = plt.figure(figsize=(20,20))
    #
    # ax = fig.add_subplot(111, title='Imaginary', projection='3d')
    #
    # coord= [0,1,2,3]
    #
    # x_pos=[]
    # y_pos=[]
    # for i in range(4):
    #     for j in range(4):
    #         x_pos.append(coord[i])
    #         y_pos.append(coord[j])
    #
    # xpos=np.array(x_pos)
    # ypos=np.array(y_pos)
    # zpos=np.array([0]*16)
    # dx = [0.6]*16
    # dy = dx
    # dz=np.squeeze(np.asarray(np.array(np.imag(den_mat).flatten())))
    #
    #
    # nrm=mpl.colors.Normalize(-1,1)
    # colors=mpl.cm.Reds(nrm(dz))
    # alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)
    #
    # for i in range(len(dx)):
    #     ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
    # xticks=['ee','eg','ge','gg']
    # yticks=xticks
    # ax.set_xticks([0.3,1.3,2.3,3.3])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0.3,1.3,2.3,3.3])
    # ax.set_yticklabels(yticks)
    # ax.set_zlim(-1,1)
    # plt.show()
    #
    # ## absolute value
    #
    # fig = plt.figure(figsize=(20,20))
    #
    # ax = fig.add_subplot(111, title='Abs', projection='3d')
    #
    # coord= [0,1,2,3]
    #
    # x_pos=[]
    # y_pos=[]
    # for i in range(4):
    #     for j in range(4):
    #         x_pos.append(coord[i])
    #         y_pos.append(coord[j])
    #
    # xpos=np.array(x_pos)
    # ypos=np.array(y_pos)
    # zpos=np.array([0]*16)
    # dx = [0.6]*16
    # dy = dx
    # dz=np.squeeze(np.asarray(np.array(np.abs(den_mat).flatten())))
    #
    #
    # nrm=mpl.colors.Normalize(-1,1)
    # colors=mpl.cm.Reds(nrm(dz))
    # alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)
    #
    # for i in range(len(dx)):
    #     ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
    # xticks=['ee','eg','ge','gg']
    # yticks=xticks
    # ax.set_xticks([0.3,1.3,2.3,3.3])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0.3,1.3,2.3,3.3])
    # ax.set_yticklabels(yticks)
    # ax.set_zlim(-1,1)
    # plt.show()

    return den_mat

def data_to_correlators(state_norm):
    IZ = (state_norm[1][0] + state_norm[3][0]) - (state_norm[0][0] + state_norm[2][0])
    ZI = (state_norm[2][0] + state_norm[3][0]) - (state_norm[0][0] + state_norm[1][0])

    IX = (state_norm[1][1] + state_norm[3][1]) - (state_norm[0][1] + state_norm[2][1])
    IY = (state_norm[1][2] + state_norm[3][2]) - (state_norm[0][2] + state_norm[2][2])

    XI = (state_norm[2][3] + state_norm[3][3]) - (state_norm[0][3] + state_norm[1][3])
    YI = (state_norm[2][6] + state_norm[3][6]) - (state_norm[0][6] + state_norm[1][6])

    XX = (state_norm[0][4] + state_norm[3][4]) - (state_norm[1][4] + state_norm[2][4])
    XY = (state_norm[0][5] + state_norm[3][5]) - (state_norm[1][5] + state_norm[2][5])
    YX = (state_norm[0][7] + state_norm[3][7]) - (state_norm[1][7] + state_norm[2][7])
    YY = (state_norm[0][8] + state_norm[3][8]) - (state_norm[1][8] + state_norm[2][8])

    ZZ = (state_norm[0][0] + state_norm[3][0]) - (state_norm[1][0] + state_norm[2][0])


    ZX = (state_norm[0][1] + state_norm[3][1]) - (state_norm[1][1] + state_norm[2][1])
    ZY = (state_norm[0][2] + state_norm[3][2]) - (state_norm[1][2] + state_norm[2][2])

    XZ = (state_norm[0][3] + state_norm[3][3]) - (state_norm[1][3] + state_norm[2][3])
    YZ = (state_norm[0][6] + state_norm[3][6]) - (state_norm[1][6] + state_norm[2][6])


    state_data = [IX,IY,IZ,XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    return state_data


def get_singleshot_data(expt_data, het_ind ,pi_cal = False):


    data_cos_list= expt_data[het_ind][0]
    data_sin_list= expt_data[het_ind][1]


    if pi_cal:

        ge_cos = np.mean(data_cos_list[-1]) - np.mean(data_cos_list[-2])
        ge_sin = np.mean(data_sin_list[-1]) - np.mean(data_sin_list[-2])

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:-2] - np.mean(data_cos_list[-2]),
                                      data_sin_list[:-2] - np.mean(data_sin_list[-2])])

        data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)


    else:
        cos_contrast = np.abs(np.max(data_cos_list)-np.min(data_cos_list))
        sin_contrast = np.abs(np.max(data_sin_list)-np.min(data_sin_list))

        if cos_contrast > sin_contrast:
            data_list = data_cos_list
        else:
            data_list = data_sin_list


    return data_cos_list, data_sin_list, data_list

def get_singleshot_data_count(expt_data, het_ind ,pi_cal = False):


    data_cos_list= expt_data[het_ind][0]
    data_sin_list= expt_data[het_ind][1]


    if pi_cal:

        ge_cos = np.mean(data_cos_list[-1]) - np.mean(data_cos_list[-2])
        ge_sin = np.mean(data_sin_list[-1]) - np.mean(data_sin_list[-2])

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:-2],
                                      data_sin_list[:-2]])

        data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)

        # plt.figure(figsize=(10,7))

        g_cos_ro = data_cos_list[-2]
        g_sin_ro = data_sin_list[-2]
        e_cos_ro = data_cos_list[-1]
        e_sin_ro = data_sin_list[-1]

        # plt.scatter(g_cos_ro,g_sin_ro)
        # plt.scatter(e_cos_ro,e_sin_ro)

        g_cos_sin_ro = np.array([g_cos_ro,g_sin_ro])
        e_cos_sin_ro = np.array([e_cos_ro,e_sin_ro])

        g_proj = np.dot(ge_mean_vec,g_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)
        e_proj = np.dot(ge_mean_vec,e_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)

#         print(np.mean(data_list[0]))
#         print(np.mean(g_proj))

        all_proj = np.array([g_proj,e_proj])
        histo_range = (all_proj.min() / 1.05, all_proj.max() * 1.05)

        g_hist, g_bins = np.histogram(g_proj,bins=1000,range=histo_range)
        e_hist, e_bins = np.histogram(e_proj,bins=1000,range=histo_range)

        g_hist_cumsum = np.cumsum(g_hist)
        e_hist_cumsum = np.cumsum(e_hist)

#         plt.figure(figsize=(7,7))
#         plt.title("qubit %s" %qubit_id)
#         plt.plot(g_bins[:-1],g_hist, 'b')
#         plt.plot(e_bins[:-1],e_hist, 'r')

        max_contrast = abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])).max()



        decision_boundary = g_bins[np.argmax(abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])))]

#         print(max_contrast)
#         print("decision boundary: %s" %decision_boundary)
#         plt.figure(figsize=(7,7))
#         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
#         plt.figure(figsize=(7,7))
#         plt.title("qubit %s" %qubit_id)
#         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
#         print(data_list.shape)

        confusion_matrix = np.array([[np.sum(g_proj<decision_boundary), np.sum(e_proj<decision_boundary)],
                                     [np.sum(g_proj>decision_boundary),np.sum(e_proj>decision_boundary)]])/data_list.shape[1]


#         print(confusion_matrix)

        confusion_matrix_inv = np.linalg.inv(confusion_matrix)

#         print(confusion_matrix_inv)

        data_count = np.array([np.sum(data_list<decision_boundary,axis=1),
                               np.sum(data_list>decision_boundary,axis=1)])/data_list.shape[1]
#         print(data_count)

        data_count_norm = np.dot(confusion_matrix_inv,data_count)

#         print(data_count_norm)

        # plt.figure(figsize=(7,7))
        # plt.title("qubit %s" %qubit_id)
        # plt.plot(data_count_norm[1])

        data_list = data_count_norm[1]

    else:
        cos_contrast = np.abs(np.max(data_cos_list)-np.min(data_cos_list))
        sin_contrast = np.abs(np.max(data_sin_list)-np.min(data_sin_list))

        if cos_contrast > sin_contrast:
            data_list = data_cos_list
        else:
            data_list = data_sin_list


    return data_cos_list, data_sin_list, data_list


def get_singleshot_data_two_qubits_4_calibration(single_data_list):

    decision_boundary_list = []
    confusion_matrix_list = []
    data_list_list = []

    for ii, expt_data in enumerate(single_data_list):

        data_cos_list= expt_data[ii][0]
        data_sin_list= expt_data[ii][1]


        if ii == 0:

            ge_cos = np.mean(data_cos_list[-2]) - np.mean(data_cos_list[-4])
            ge_sin = np.mean(data_sin_list[-2]) - np.mean(data_sin_list[-4])
        elif ii == 1:
            ge_cos = np.mean(data_cos_list[-3]) - np.mean(data_cos_list[-4])
            ge_sin = np.mean(data_sin_list[-3]) - np.mean(data_sin_list[-4])

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:],
                                      data_sin_list[:]])

        data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)

        data_list_list.append(data_list)

        # plt.figure(figsize=(7,7))
        if ii == 0:
            g_cos_ro = data_cos_list[-4]
            g_sin_ro = data_sin_list[-4]
            e_cos_ro = data_cos_list[-2]
            e_sin_ro = data_sin_list[-2]
        elif ii == 1:
            g_cos_ro = data_cos_list[-4]
            g_sin_ro = data_sin_list[-4]
            e_cos_ro = data_cos_list[-3]
            e_sin_ro = data_sin_list[-3]


        # plt.scatter(g_cos_ro,g_sin_ro)
        # plt.scatter(e_cos_ro,e_sin_ro)

        g_cos_sin_ro = np.array([g_cos_ro,g_sin_ro])
        e_cos_sin_ro = np.array([e_cos_ro,e_sin_ro])

        g_proj = np.dot(ge_mean_vec,g_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)
        e_proj = np.dot(ge_mean_vec,e_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)

#         print(np.mean(data_list[0]))
#         print(np.mean(g_proj))

        all_proj = np.array([g_proj,e_proj])
        histo_range = (all_proj.min() / 1.05, all_proj.max() * 1.05)

        g_hist, g_bins = np.histogram(g_proj,bins=1000,range=histo_range)
        e_hist, e_bins = np.histogram(e_proj,bins=1000,range=histo_range)

        g_hist_cumsum = np.cumsum(g_hist)
        e_hist_cumsum = np.cumsum(e_hist)

#         plt.figure(figsize=(7,7))
# #             plt.title("qubit %s" %qubit_id)
#         plt.plot(g_bins[:-1],g_hist, 'b')
#         plt.plot(e_bins[:-1],e_hist, 'r')

        max_contrast = abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])).max()



        decision_boundary = g_bins[np.argmax(abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])))]

        decision_boundary_list.append(decision_boundary)

        # print(max_contrast)
#         print("decision boundary: %s" %decision_boundary)
#         plt.figure(figsize=(7,7))
#         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
#         plt.figure(figsize=(7,7))
#         plt.title("qubit %s" %qubit_id)
#         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
#         print(data_list.shape)

        confusion_matrix = np.array([[np.sum(g_proj<decision_boundary), np.sum(e_proj<decision_boundary)],
                                     [np.sum(g_proj>decision_boundary),np.sum(e_proj>decision_boundary)]])/data_list.shape[1]

        confusion_matrix_list.append(confusion_matrix)
#         print(confusion_matrix)

        confusion_matrix_inv = np.linalg.inv(confusion_matrix)

#         print(confusion_matrix_inv)

        data_count = np.array([np.sum(data_list<decision_boundary,axis=1),
                               np.sum(data_list>decision_boundary,axis=1)])/data_list.shape[1]
#         print(data_count)

        data_count_norm = np.dot(confusion_matrix_inv,data_count)

#         print(data_count_norm)
#
#         plt.figure(figsize=(7,7))
# #             plt.title("population by counting: qubit %s" %qubit_id)
#         plt.plot(data_count_norm[1])

        data_list = data_count_norm[1]

    # print(decision_boundary_list)
    # print(confusion_matrix_list)
    gg = np.sum(np.bitwise_and((data_list_list[0] < decision_boundary_list[0]) ,
                               (data_list_list[1] < decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    ge = np.sum(np.bitwise_and((data_list_list[0] < decision_boundary_list[0]) ,
                               (data_list_list[1] > decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    eg = np.sum(np.bitwise_and((data_list_list[0] > decision_boundary_list[0]) ,
                               (data_list_list[1] < decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    ee = np.sum(np.bitwise_and((data_list_list[0] > decision_boundary_list[0]) ,
                               (data_list_list[1] > decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]

    total_confusion_matrix = np.kron(confusion_matrix_list[0],confusion_matrix_list[1])

    ## 4- confusion
    # print("4confusion")
    # print(len(data_list_list[0]))
    gg_confusion = [np.sum(np.bitwise_and((data_list_list[0][kk] < decision_boundary_list[0]) ,
                    (data_list_list[1][kk] < decision_boundary_list[1])))/data_list_list[0].shape[1] for kk in range(-4,0)]
    ge_confusion = [np.sum(np.bitwise_and((data_list_list[0][kk] < decision_boundary_list[0]) ,
                    (data_list_list[1][kk] > decision_boundary_list[1])))/data_list_list[0].shape[1] for kk in range(-4,0)]
    eg_confusion = [np.sum(np.bitwise_and((data_list_list[0][kk] > decision_boundary_list[0]) ,
                    (data_list_list[1][kk] < decision_boundary_list[1])))/data_list_list[0].shape[1] for kk in range(-4,0)]
    ee_confusion = [np.sum(np.bitwise_and((data_list_list[0][kk] > decision_boundary_list[0]) ,
                    (data_list_list[1][kk] > decision_boundary_list[1])))/data_list_list[0].shape[1] for kk in range(-4,0)]
    # print(gg_confusion)

    four_confusion = np.array([gg_confusion, ge_confusion, eg_confusion, ee_confusion])

    # print(four_confusion)

#     print(np.dot(np.linalg.inv(total_confusion_matrix),total_confusion_matrix))

    total_confusion_matrix_inv = np.linalg.inv(four_confusion)
    state = np.array([gg[:-4],ge[:-4],eg[:-4],ee[:-4]])

    # print(total_confusion_matrix)
    # print(total_confusion_matrix_inv)

    state_norm = np.dot(total_confusion_matrix_inv,state)

    return state_norm

#     print(state_norm.shape)

    # plt.figure(figsize=(7,7))
    # plt.plot(gg,label='gg')
    # plt.plot(ge,label='ge')
    # plt.plot(eg,label='eg')
    # plt.plot(ee,label='ee')
    #
    # plt.legend()
    #
    #
    # plt.figure(figsize=(7,7))
    # plt.plot(state_norm[0],label='gg')
    # plt.plot(state_norm[1],label='ge')
    # plt.plot(state_norm[2],label='eg')
    # plt.plot(state_norm[3],label='ee')
    #
    # plt.legend()



def get_singleshot_data_two_qubits(single_data_list,pi_cal = False):

    decision_boundary_list = []
    confusion_matrix_list = []
    data_list_list = []

    for ii, expt_data in enumerate(single_data_list):

        data_cos_list= expt_data[ii][0]
        data_sin_list= expt_data[ii][1]


        if pi_cal:

            ge_cos = np.mean(data_cos_list[-1]) - np.mean(data_cos_list[-2])
            ge_sin = np.mean(data_sin_list[-1]) - np.mean(data_sin_list[-2])

            ge_mean_vec = np.array([ge_cos,ge_sin])

            data_cos_sin_list = np.array([data_cos_list[:-2],
                                          data_sin_list[:-2]])

            data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


            data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)

            data_list_list.append(data_list)

            # plt.figure(figsize=(10,7))

            g_cos_ro = data_cos_list[-2]
            g_sin_ro = data_sin_list[-2]
            e_cos_ro = data_cos_list[-1]
            e_sin_ro = data_sin_list[-1]

            # plt.scatter(g_cos_ro,g_sin_ro)
            # plt.scatter(e_cos_ro,e_sin_ro)

            g_cos_sin_ro = np.array([g_cos_ro,g_sin_ro])
            e_cos_sin_ro = np.array([e_cos_ro,e_sin_ro])

            g_proj = np.dot(ge_mean_vec,g_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)
            e_proj = np.dot(ge_mean_vec,e_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)

    #         print(np.mean(data_list[0]))
    #         print(np.mean(g_proj))

            all_proj = np.array([g_proj,e_proj])
            histo_range = (all_proj.min() / 1.05, all_proj.max() * 1.05)

            g_hist, g_bins = np.histogram(g_proj,bins=1000,range=histo_range)
            e_hist, e_bins = np.histogram(e_proj,bins=1000,range=histo_range)

            g_hist_cumsum = np.cumsum(g_hist)
            e_hist_cumsum = np.cumsum(e_hist)

    #         plt.figure(figsize=(7,7))
    #         plt.title("qubit %s" %qubit_id)
    #         plt.plot(g_bins[:-1],g_hist, 'b')
    #         plt.plot(e_bins[:-1],e_hist, 'r')

            max_contrast = abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])).max()



            decision_boundary = g_bins[np.argmax(abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])))]

            decision_boundary_list.append(decision_boundary)

    #         print(max_contrast)
    #         print("decision boundary: %s" %decision_boundary)
    #         plt.figure(figsize=(7,7))
    #         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
    #         plt.figure(figsize=(7,7))
    #         plt.title("qubit %s" %qubit_id)
    #         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
    #         print(data_list.shape)

            confusion_matrix = np.array([[np.sum(g_proj<decision_boundary), np.sum(e_proj<decision_boundary)],
                                         [np.sum(g_proj>decision_boundary),np.sum(e_proj>decision_boundary)]])/data_list.shape[1]

            confusion_matrix_list.append(confusion_matrix)
    #         print(confusion_matrix)

            confusion_matrix_inv = np.linalg.inv(confusion_matrix)

    #         print(confusion_matrix_inv)

            data_count = np.array([np.sum(data_list<decision_boundary,axis=1),
                                   np.sum(data_list>decision_boundary,axis=1)])/data_list.shape[1]
    #         print(data_count)

            data_count_norm = np.dot(confusion_matrix_inv,data_count)

    #         print(data_count_norm)

            # plt.figure(figsize=(7,7))
            # plt.title("qubit %s" %qubit_id)
            # plt.plot(data_count_norm[1])

            data_list = data_count_norm[1]

    # print(decision_boundary_list)
    # print(confusion_matrix_list)
    gg = np.sum(np.bitwise_and((data_list_list[0] < decision_boundary_list[0]) ,
                               (data_list_list[1] < decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    ge = np.sum(np.bitwise_and((data_list_list[0] < decision_boundary_list[0]) ,
                               (data_list_list[1] > decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    eg = np.sum(np.bitwise_and((data_list_list[0] > decision_boundary_list[0]) ,
                               (data_list_list[1] < decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    ee = np.sum(np.bitwise_and((data_list_list[0] > decision_boundary_list[0]) ,
                               (data_list_list[1] > decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]

    total_confusion_matrix = np.kron(confusion_matrix_list[0],confusion_matrix_list[1])

    total_confusion_matrix_inv = np.linalg.inv(total_confusion_matrix)
    state = np.array([gg,ge,eg,ee])

    state_norm = np.dot(total_confusion_matrix_inv,state)

    return state_norm

def get_iq_data(expt_data,het_freq = 0.148, td=0, pi_cal = False):
    data_cos_list=[]
    data_sin_list=[]

    alazar_time_pts = np.arange(len(np.array(expt_data)[0]))

    for data in expt_data:
        cos = np.cos(2*np.pi*het_freq*(alazar_time_pts-td))
        sin = np.sin(2*np.pi*het_freq*(alazar_time_pts-td))

        data_cos = np.dot(data,cos)/len(cos)
        data_sin = np.dot(data,sin)/len(sin)

        data_cos_list.append(data_cos)
        data_sin_list.append(data_sin)

    data_cos_list = np.array(data_cos_list)
    data_sin_list = np.array(data_sin_list)

    if pi_cal:

        ge_cos = data_cos_list[-1] - data_cos_list[-2]
        ge_sin = data_sin_list[-1] - data_sin_list[-2]

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:-2] - data_cos_list[-2],data_sin_list[:-2] - data_sin_list[-2]])

        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)


    else:
        cos_contrast = np.abs(np.max(data_cos_list)-np.min(data_cos_list))
        sin_contrast = np.abs(np.max(data_sin_list)-np.min(data_sin_list))

        if cos_contrast > sin_contrast:
            data_list = data_cos_list
        else:
            data_list = data_sin_list

    return data_cos_list, data_sin_list, data_list



Nfeval = 0
Xeval = []
Cost = []


def density_matrix_maximum_likelihood(m_ab, input_guess):

    # Making Xguess equal to perfect bell
#     perfect_bell = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
#     perfect_bell_den_mat = np.outer(perfect_bell, perfect_bell)

#     Xguess_real = perfect_bell_den_mat
#     Xguess_imag = np.zeros((4,4))

    Xguess_real = np.real(input_guess)
    Xguess_imag = np.imag(input_guess)

    Xguess = np.vstack((Xguess_real,Xguess_imag))




    # Set boundary for allocation of each X to be between -1 and 1, between 0 and 1 if on diagonal
    bnds = []
    for m_ind in ['real','imag']:
        for ii in range(4):
            for jj in range(4):
                if ii == jj:
                    bnds.append((np.long(-1),np.long(1)))
                else:
                    bnds.append((np.long(-1),np.long(1)))




    I = np.matrix([[1,0],[0,1]])
    X = np.matrix([[0,1],[1,0]])
    Y = np.matrix([[0,-1j],[1j,0]])
    Z = np.matrix([[1,0],[0,-1]])
    def get_Rx():
        theta = np.pi/2
        return np.cos(theta/2) * I -1j*np.sin(theta/2)*X

    def get_Ry():
        theta = np.pi/2
        return np.cos(theta/2) * I -1j*np.sin(theta/2)*Y

    def get_nRx():
        theta = -np.pi/2
        return np.cos(theta/2) * I -1j*np.sin(theta/2)*X

    def get_nRy():
        theta = -np.pi/2
        return np.cos(theta/2) * I -1j*np.sin(theta/2)*Y

    def get_identity():
        return I


    def convert_array_to_matrix(x_array):
        x_real = x_array[0:16]
        x_imag = np.multiply(x_array[16:32],1j)
        x = x_real+x_imag
        x = np.reshape(x,(4,4))
        return x

    measurement_pulse = [['I','I'], ['I','X'],['I','Y'],['X','I'],['X','X'],['X','Y'],['Y','I'],['Y','X'],['Y','Y'],
                             ['I','-X'],['I','-Y'],['-X','I'],['-X','-X'],['-X','-Y'],['-Y','I'],['-Y','-X'],['-Y','-Y']]


    R_dict = {'I':get_identity, 'X':get_Rx, 'Y': get_Ry, '-X':get_nRx, '-Y':get_nRy}

    def get_rotation(m_pulse):

        return np.kron(R_dict[m_pulse[0]](),R_dict[m_pulse[1]]())

    def get_eigenvalues(x_array):
        x = convert_array_to_matrix(x_array)
        ew, ev = np.linalg.eigh(x)

        return np.real(ew)

    def get_trace(x_array):
        x = convert_array_to_matrix(x_array)
        return np.real(np.trace(x))


    def get_conjugate_diff(x_array):
        x = convert_array_to_matrix(x_array)
        xt = np.transpose(np.conjugate(x))

        return np.sum(np.square(x-xt))


    cons_list = []
    cons_list.append({'type': 'ineq', 'fun': lambda x: get_eigenvalues(x)[0]})
    cons_list.append({'type': 'ineq', 'fun': lambda x: get_eigenvalues(x)[1]})
    cons_list.append({'type': 'ineq', 'fun': lambda x: get_eigenvalues(x)[2]})
    cons_list.append({'type': 'ineq', 'fun': lambda x: get_eigenvalues(x)[3]})
    cons_list.append({'type': 'eq', 'fun': lambda x: get_trace(x)-1})
#     cons_list.append({'type': 'eq', 'fun': lambda x: get_conjugate_diff(x)-1})
    cons = tuple(cons_list)

    # error function
    def error_function(x_array):

        x = convert_array_to_matrix(x_array)

        err = 0

        for ii, m_pulse in enumerate(measurement_pulse):
            rotation_pulse = get_rotation(m_pulse)

            Ut_rho_U = np.dot(np.dot(np.linalg.inv(rotation_pulse),x),rotation_pulse)
            for jj in range(4):
                diff = Ut_rho_U[jj,jj] - m_ab[jj,ii]
                err += np.absolute(diff)**2

        return err



    def callbackF(Xi):
        global Nfeval
        Nfeval += 1

        cost = error_function(Xi)
        # print(str(Nfeval)+ ': cost function: ' + str(cost))

        global Xeval
        Xeval.append(Xi)

        global Cost
        Cost.append(cost)



    min_result = spo.minimize(error_function, Xguess, method='SLSQP', options={'disp': False,'maxiter':500}
                              , bounds=bnds,constraints=cons,callback=callbackF)

    # optimize allocation and Sharpe ratio
    optimized_fun = min_result.fun
    optimized_x = min_result.x

    #print optimized_x
    return convert_array_to_matrix(optimized_x)


