from scipy.interpolate import interp1d
import numpy as np

# Geophone calibration curve data

def volts_to_velocity(frequency, volts):
    """
    :param frequency: Frequency in Hz
    :param volts: Voltage at that particular frequency
    :return: Velocity in m/s estimated from data.
    """

    # Data from the website. 1st column: Freq (Hz); 2nd column V/(in/s)
    data = np.array([[1.6, 0.11],
                     [2.0, 0.19],
                     [2.5, 0.30],
                     [3.0, 0.50],
                     [3.5, 0.72],
                     [4.0, 1.00],
                     [4.5, 1.20],
                     [5.2, 1.32],
                     [6.0, 1.20],
                     [8.0, 1.04],
                     [10.0, 0.95],
                     [12.0, 0.91],
                     [16.0, 0.87],
                     [20.0, 0.85],
                     [40.0, 0.82],
                     [100, 0.81]])

    interpolated_data = interp1d(data[:,0], data[:,1], kind='cubic')

    if len(frequency)>1 and len(frequency)==len(volts):
        velocity=[]
        for idx,F in enumerate(frequency):
            velocity.append(volts/interpolated_data(F)*25.4E-3)
        velocity = np.array(velocity)
    elif len(frequency)==len(volts):
        velocity = volts/interpolated_data(frequency)*25.4E-3

    return velocity


def velocity_to_displacement(frequency, velocity):
    """
    :param frequency: Frequency in Hz
    :param velocity: Velocity in m/s
    :return: Displacement in m
    """
    displacement = velocity/(2*np.pi*frequency)
    return displacement

def get_displacement(fpoints, volts):

    velocity = volts_to_velocity(fpoints, volts)
    displacement = velocity_to_displacement(fpoints, velocity)
    return displacement