from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit


def gauss(x, mu, sigma, a):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


def bimodal(x,mu1,sigma1, a1, mu2, sigma2, a2):
    return gauss(x, mu1, sigma1, a1)+gauss(x, mu2, sigma2, a2)


def calc_demarcation(di, dq, isPlot=False):
    # Use a K-Means Cluster to automatically assign points in IQ space, with 2 total clusters
    clf = KMeans(n_clusters=2)
    data = np.array(list(zip(di, dq)))
    clf.fit(data)
    # Find the centroids of each cluster
    centers = clf.cluster_centers_
    labels = clf.predict(data)

    # Determine the perpendicular slope to the line that connects the two centroids
    slope = ((centers[1][1] - centers[0][1]) / (centers[1][0] - centers[0][0]))
    angle = - np.arctan(slope)

    if np.abs(slope) > 0.1:
        slope = -1 / slope

    # Find the midpoint of the two centroids.
    if centers[1][0] > centers[0][0]:
        midpointX = (centers[1][0] - centers[0][0]) / 2
    else:
        midpointX = (centers[0][0] - centers[1][0]) / 2

    if centers[1][1] > centers[0][1]:
        midpointY = (centers[1][1] - centers[0][1]) / 2
    else:
        midpointY = (centers[0][1] - centers[1][1]) / 2

    # Find the line that goes through the midpoint of the centroids and has the perpendicular slope
    intercept = midpointY - slope * midpointX

    if isPlot:
        plt.close()
        plt.scatter(di, dq, c=labels)
        plt.scatter(centers[:, 0], centers[:, 1])
        plt.show()

    return slope, intercept, angle


def get_num_tau(signal, u_thresh_up, u_thresh_dn, params):
    delta_t, delta_u, A_dn, A_up = params

    # Determine the number of transitions in the data
    k = 0
    state = 0  # Initial state is down
    classified = []

    # Loop through each point of the signal
    for i in range(len(signal)):
        if state:
            # If the state is currently up, and new point is below the threshold, flip the state and increment k
            if signal[i] < u_thresh_up:
                state = 0
                k = k + 1
                classified.append(0)
            else:
                classified.append(1)
        else:
            # If the state is currently down, and new point is above the threshold, flip the state and increment k
            if signal[i] > u_thresh_dn:
                state = 1
                k = k + 1
                classified.append(1)
            else:
                classified.append(0)

    # If there is a bad fit/other irregularity, set minimum number of transitions to be 10
    if k < 10:
        k = 10

    # Using the number of transitions, determine the lifetimes of each state (in us). Uses the limit where k is large
    tau_dn = (2 * A_dn * delta_t) / (k * delta_u)
    tau_up = (2 * A_up * delta_t) / (k * delta_u)

    return k, tau_dn, tau_up, classified


def get_thresholds(signal, avgNum, expected, delta_t):
    # Average signal
    averagedSignal = []
    for i in range(len(signal) - avgNum):
        averagedSignal.append(np.average(signal[i:i + avgNum]))

    # Convert data into histogram and fit to bimodal gaussian
    n, bins = np.histogram(averagedSignal, bins=80)
    x = (bins[1:] + bins[:-1]) / 2

    #     expected = (25, 100, 70, 1300, 100, 50)
    params, cov = curve_fit(bimodal, x, n, expected)

    # Estimate initial number of transitions

    delta_u = bins[1] - bins[0]  # Determine bin size

    # Extract fit parameters from the double gaussian curve
    if params[0] < params[3]:
        u_dn = params[0]
        sigma_dn = params[1]
        A_dn = params[2] * np.abs(sigma_dn) * np.sqrt(2 * np.pi)
        u_up = params[3]
        sigma_up = params[4]
        A_up = params[5] * np.abs(sigma_up) * np.sqrt(2 * np.pi)
    else:
        u_up = params[0]
        sigma_up = params[1]
        A_up = params[2] * np.abs(sigma_up) * np.sqrt(2 * np.pi)
        u_dn = params[3]
        sigma_dn = params[4]
        A_dn = params[5] * np.abs(sigma_dn) * np.sqrt(2 * np.pi)

    # From the curve, the separation between the two states
    delta_v = u_up - u_dn

    # Initial overestimate of threshold positions
    u_thresh_dn_0 = u_dn + delta_v / 2 + sigma_dn ** 2 / delta_u
    u_thresh_up_0 = u_up + delta_v / 2 - sigma_up ** 2 / delta_u

    # First estimate of num transitions and lifetimes
    k_current, tau_dn, tau_up, classified = get_num_tau(averagedSignal, u_thresh_up_0, u_thresh_dn_0,
                                                        [delta_t, delta_u, A_dn, A_up])

    # Iteratively calculate num of transitions, lifetimes, and threshold values, until number of transitions converges
    i = 0
    k_new = 10000

    u_thresh_dn_1 = 0
    u_thresh_up_1 = 0

    while abs(k_new - k_current) > 5 and i < 1000:
        k_current = k_new
        i = i + 1

        # Get new threshold values using the current estimates of lifetimes and number of transitions
        delta_up_1 = delta_v / 2 + sigma_up ** 2 / delta_v * np.log(tau_up / delta_t - 1)
        delta_dn_1 = delta_v / 2 + sigma_dn ** 2 / delta_v * np.log(tau_dn / delta_t - 1)

        u_thresh_dn_1 = u_dn + delta_dn_1
        u_thresh_up_1 = u_up - delta_up_1

        # Recalculate the number of transitions and lifetimes
        k_new, tau_dn, tau_up, classified = get_num_tau(averagedSignal, u_thresh_up_1, u_thresh_dn_1,
                                                        [delta_t, delta_u, A_dn, A_up])

        # Recalculate the number of points in the up and down Gaussian
        A_up = np.sum(classified) * delta_u
        A_dn = len(classified) * delta_u - A_up

    print("number of iterations: " + str(i))
    return k_new, tau_dn, tau_up, u_thresh_dn_1, u_thresh_up_1, classified


fileNumber = 0
expt_name = "ContinuousReadout"
filename = "S:\\_data\\210719 - DoubleFluxonium_S3C3\\data\\" + str(fileNumber).zfill(
    5) + "_" + expt_name.lower() + ".h5"
with File(filename, 'r') as a:
    di1 = array(a['di1'])
    dq1 = array(a['dq1'])

slope, intercept, angle = calc_demarcation(di1[0], dq1[0])

signal = di1[0] * slope - dq1[0] + intercept

#Get the histogram to estimate initial fitting parameters

n, bins = np.histogram(signal, bins = 80)
amp = np.max(n)
peak1 = bins[np.argmax(n)]

numAvg = 1  # If you want to do a rolling average over more data points
k, tau_dn, tau_up, u_thresh_dn, u_thresh_up, classified = get_thresholds(signal, numAvg,
                                 expected= (peak1 - 1000, 100, amp, peak1, 100, amp), delta_t=3.6)
print(angle)
print(u_thresh_dn)  # The number to compare to for ground/excited
