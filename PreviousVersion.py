import datetime
import smbus
from scipy import fftpack
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# select the correct i2c bus for this revision of Raspberry Pi
revision = ([l[12:-1] for l in open('/proc/cpuinfo', 'r').readlines() if l[:8] == "Revision"] + ['0000'])[0]
bus = smbus.SMBus(1 if int(revision, 16) >= 4 else 0)

# ADXL345 constants
EARTH_GRAVITY_MS2 = 9.80665
SCALE_MULTIPLIER = 0.004

DATA_FORMAT = 0x31
BW_RATE = 0x2C
POWER_CTL = 0x2D

BW_RATE_1600HZ = 0x0F
BW_RATE_800HZ = 0x0E
BW_RATE_400HZ = 0x0D
BW_RATE_200HZ = 0x0C
BW_RATE_100HZ = 0x0B
BW_RATE_50HZ = 0x0A
BW_RATE_25HZ = 0x09

RANGE_2G = 0x00
RANGE_4G = 0x01
RANGE_8G = 0x02
RANGE_16G = 0x03

MEASURE = 0x08
AXES_DATA = 0x32

# other constants

samples_per_second = 10;
time_for_sample = 1.0 / samples_per_second
archive_txt = "textfile_" + datetime.datetime.now().strftime("%d-%m-%Y__%H_%M_%S") + ".txt"
archive_csv_rd = "rd_csvfile_" + datetime.datetime.now().strftime("%d-%m-%Y__%H_%M_%S") + ".csv"
archive_csv_fft = "fft_csvfile_" + datetime.datetime.now().strftime("%d-%m-%Y__%H_%M_%S") + ".csv"

# samples_to_read = 10000
sample_rate = 1030
T = 1.0 / sample_rate

channel_1 = []
channel_2 = []
channel_3 = []


#####functions#####
def conv_str_tag(channel, tag):
    # Convert every channel from int to str, separated by a coma and adds tags at the beginning and end.
    n = len(channel)
    s_channel = '<' + tag + '>'
    for i in range(n - 1):
        s_channel = s_channel + str(channel[i]) + ','
    s_channel = s_channel + str(channel[n - 1]) + '</' + tag + '>'
    return s_channel


#####Add tags and save on file#####
def save(channel_1, channel_2, channel_3, archive):
    str_channel = ''
    str_channel += conv_str_tag(channel_1, 'L1') + '\n'
    str_channel += conv_str_tag(channel_2, 'L2') + '\n'
    str_channel += conv_str_tag(channel_3, 'L3') + '\n'

    # Write to file
    arch = open("/opt/projects/VibroNoiseAnalyzer/snmahajan//recordings" + archive, "w")
    arch.write(str_channel)
    arch.close()


class ADXL345:
    address = None

    def __init__(self, address=0x53):
        self.address = address
        self.setBandwidthRate(BW_RATE_1600HZ)
        self.setRange(RANGE_16G)
        self.enableMeasurement()

    def enableMeasurement(self):
        bus.write_byte_data(self.address, POWER_CTL, MEASURE)

    def setBandwidthRate(self, rate_flag):
        bus.write_byte_data(self.address, BW_RATE, rate_flag)

    # set the measurement range for 10-bit readings
    def setRange(self, range_flag):
        value = bus.read_byte_data(self.address, DATA_FORMAT)

        value &= ~0x0F;
        value |= range_flag;
        value |= 0x08;

        bus.write_byte_data(self.address, DATA_FORMAT, value)

    # returns the current reading from the sensor for each axis
    #
    # parameter gforce:
    #    False (default): result is returned in m/s^2
    #    True           : result is returned in gs
    def getAxes(self, gforce=False):
        bytes = bus.read_i2c_block_data(self.address, AXES_DATA, 6)

        x = bytes[0] | (bytes[1] << 8)
        if (x & (1 << 16 - 1)):
            x = x - (1 << 16)

        y = bytes[2] | (bytes[3] << 8)
        if (y & (1 << 16 - 1)):
            y = y - (1 << 16)

        z = bytes[4] | (bytes[5] << 8)
        if (z & (1 << 16 - 1)):
            z = z - (1 << 16)

        x = x * SCALE_MULTIPLIER
        y = y * SCALE_MULTIPLIER
        z = z * SCALE_MULTIPLIER

        if gforce == False:
            x = x * EARTH_GRAVITY_MS2
            y = y * EARTH_GRAVITY_MS2
            z = z * EARTH_GRAVITY_MS2

        x = round(x, 4)
        y = round(y, 4)
        z = round(z, 4)

        return {"x": x, "y": y, "z": z}


class View:
    def __init__(self):
        fig1 = plt.figure(num=1, figsize=(10, 7))
        fig1.suptitle('Sampled signal - Acceleration')

        # Figure 1. Sampled signals.

        # Channel X
        self.ssaX = fig1.add_subplot(3, 1, 1)
        self.ssaX.set_title("Channel X")
        self.ssaX.set_xlabel('ms')
        self.ssaX.set_ylabel('g')
        self.ssaX.grid()

        # Channel Y
        self.ssaY = fig1.add_subplot(3, 1, 2)
        self.ssaY.set_title("Channel Y")
        self.ssaY.set_xlabel('ms')
        self.ssaY.set_ylabel('g')
        self.ssaY.grid()

        # Channel Z
        self.ssaZ = fig1.add_subplot(3, 1, 3)
        self.ssaZ.set_title("Channel Z")
        self.ssaZ.set_xlabel('ms')
        self.ssaZ.set_ylabel('g')
        self.ssaZ.grid()

        # Figure 2. FFT from signals.
        fig2 = plt.figure(num=2, figsize=(10, 7))
        fig2.suptitle('FFT spectrum')

        # Channel X
        self.ssaXF = fig2.add_subplot(3, 1, 1)
        self.ssaXF.grid()
        self.ssaXF.set_title("Channel X")
        self.ssaXF.set_xlabel('Hz')
        self.ssaXF.set_ylabel('g')

        # Channel Y
        self.ssaYF = fig2.add_subplot(3, 1, 2)
        self.ssaYF.grid()
        self.ssaYF.set_title("Channel Y")
        self.ssaYF.set_xlabel('Hz')
        self.ssaYF.set_ylabel('g')

        # Channel Z
        self.ssaZF = fig2.add_subplot(3, 1, 3)
        self.ssaZF.grid()
        self.ssaZF.set_title("Channel Z")
        self.ssaZF.set_xlabel('Hz')
        self.ssaZF.set_ylabel('g')

        plt.ion()

    def show_graph(self, channel_1, channel_2, channel_3, channel_1_fft, channel_2_fft, channel_3_fft):
        stime = len(channel_1) / 1000
        num_data = len(channel_1)
        x = np.linspace(0, 1000, num=num_data)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), int(num_data / 2))
        N = len(channel_1)

        # self.ssaX.cla()
        self.ssaX.plot(x, channel_1)
        self.ssaY.plot(x, channel_2)
        self.ssaZ.plot(x, channel_3)
        self.ssaXF.plot(xf, np.abs(channel_1_fft[:int(N / 2)]))
        self.ssaYF.plot(xf, np.abs(channel_2_fft[:int(N / 2)]))
        self.ssaZF.plot(xf, np.abs(channel_3_fft[:int(N / 2)]))
        # plt.pause(0.25)

        plt.show()
        plt.pause(0.25)


def make_record():
    global channel_1
    global channel_2
    global channel_3

    # print("Amount of samples in channel 1: %s" %len(channel_1))
    # print("Amount of samples in channel 2: %s" %len(channel_2))
    # print("Amount of samples in channel 3: %s" %len(channel_3))

    #####saving to TXT file#####
    # print("Saving to %s" %archive)
    # save(channel_1, channel_2, channel_3, archive)

    #####Calculate average value for each channel#####
    num_data = len(channel_1)
    X = range(0, num_data, 1)
    vdc_channel_1 = 0
    vdc_channel_2 = 0
    vdc_channel_3 = 0
    for indice in X:
        vdc_channel_1 += channel_1[indice]
        vdc_channel_2 += channel_2[indice]
        vdc_channel_3 += channel_3[indice]
    vdc_channel_1 = vdc_channel_1 / num_data
    vdc_channel_2 = vdc_channel_2 / num_data
    vdc_channel_3 = vdc_channel_3 / num_data

    # print("Vdc Channel 1: ",vdc_channel_1)
    # print("Vdc Channel 2: ",vdc_channel_2)
    # print("Vdc Channel 3: ",vdc_channel_3)

    #####Subtract DC offset#####
    for indice in X:
        channel_1[indice] -= vdc_channel_1
        channel_2[indice] -= vdc_channel_2
        channel_3[indice] -= vdc_channel_3

    #####saving to CSV file#####
    arch = open(archive_csv_rd, "w")
    arch.truncate(0)
    num_data = len(channel_1)
    indice = 0;
    while (indice < num_data):
        arch.write(str(channel_1[indice]) + "," + str(channel_2[indice]) + "," + str(channel_3[indice]) + "\n")
        indice = indice + 1;

    arch.close()
    print("Saving complete")

    #####calculation of fft#####
    channel_fft_x = []
    channel_fft_y = []
    channel_fft_z = []

    N = len(channel_1)  # length of the signal

    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    yf1 = fftpack.fft(channel_1)
    channel_fft_x = 2.0 / N * np.abs(yf1[:int(N / 2)])

    yf2 = fftpack.fft(channel_2)
    channel_fft_y = 2.0 / N * np.abs(yf2[:int(N / 2)])

    yf3 = fftpack.fft(channel_3)
    channel_fft_z = 2.0 / N * np.abs(yf3[:int(N / 2)])

    #####saving to CSV file#####
    arch = open(archive_csv_fft, "w")
    arch.truncate(0)
    num_data = len(xf)
    indice = 0;
    while (indice < num_data):
        arch.write(str(xf[indice]) + "," + str(channel_fft_x[indice]) + "," + str(channel_fft_y[indice]) + "," + str(
            channel_fft_z[indice]) + "\n")
        indice = indice + 1;

    arch.close()

    # view.show_graph(channel_1, channel_2, channel_3, channel_fft_x, channel_fft_y, channel_fft_z)

    channel_1 = []
    channel_2 = []
    channel_3 = []

    print("CYCLE PASSED")


def mainprog():
    adxl345 = ADXL345()
    print("START " + str(time.time()))

    startTime = time.time()
    lastCheckTime = startTime
    nextTime = lastCheckTime + time_for_sample

    # plt.ion()
    # view = View()

    while (time.time() < startTime + 10.0):
        axes = adxl345.getAxes(True)  # False = m/s^2, True = g
        # put the axes into variables
        x = axes['x']
        y = axes['y']
        z = axes['z']

        channel_1.append(x)
        channel_2.append(y)
        channel_3.append(z)
        # print("A")
        if (time.time() >= nextTime):
            lastCheckTime = nextTime
            nextTime = lastCheckTime + time_for_sample

            make_record()


if __name__ == "__main__":
    mainprog()
