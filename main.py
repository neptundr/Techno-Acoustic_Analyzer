import time
import smbus
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import os

# select the correct i2c bus for this revision of Raspberry Pi
revision = ([l[12:-1] for l in open('/proc/cpuinfo', 'r').readlines() if l[:8] == "Revision"] + ['0000'])[0]
bus = smbus.SMBus(1 if int(revision, 16) >= 4 else 0)

# region ADXL345 constants
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

# endregion

# region Operating constants
slice_time = 25  # the number of seconds taken for one measure
duration = 3
slice_not_time = True  # if True duration stands for number of slices taken, otherwise - minimun recording duration in seconds; will be removed
record_reference_standard = False
save_graphs = True

number_of_channels = 3  # audio will serve as the fourth channel
sample_rate = 1030
path = os.path.abspath(__file__)[0: os.path.abspath(__file__).rfind("/")] + '/N1/'

# endregion


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

        value &= ~0x0F
        value |= range_flag
        value |= 0x08

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

        if not gforce:
            x = x * EARTH_GRAVITY_MS2
            y = y * EARTH_GRAVITY_MS2
            z = z * EARTH_GRAVITY_MS2

        x = round(x, 4)
        y = round(y, 4)
        z = round(z, 4)

        return [x, y, z]


class Channel:
    def __init__(self):
        self.channel = []
        self.channel_fft = []

    def add_data(self, data):
        self.channel.append(data)

    def reset(self):
        self.channel = []
        self.channel_fft = []

    def subtract_average(self):
        sum = 0
        for i in range(len(self.channel)):
            sum += self.channel[i]
        average = sum / len(self.channel)

        for i in range(len(self.channel)):
            self.channel[i] -= average

    def calculate_fft(self):
        self.channel_fft = 2.0 / len(self.channel) * np.abs(fftpack.fft(self.channel)[:len(self.channel) // 2])

    def save_graphs(self, current_slice, signal_type):

        stime = sample_rate / 1000
        x_seq_signal = np.linspace(0, len(self.channel) / stime, num=len(self.channel))
        fig = plt.figure(num=1, figsize=(10, 7))
        fig.suptitle(signal_type)

        # signal
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(x_seq_signal, self.channel)
        ax.set_title("Signal")
        ax.set_xlabel('ms')
        ax.set_ylabel('g')
        ax.grid()

        # fft
        T = 1.0 / sample_rate
        x_seq_fft = np.linspace(0.0, 1.0 / (2.0 * T), len(self.channel) // 2)

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(x_seq_fft, 2.0 / len(self.channel) * np.abs(self.channel_fft[:len(self.channel) // 2]))
        ax.grid()
        ax.set_title("FFT spectrum")
        ax.set_xlabel('Hz')
        ax.set_ylabel('g')

        plt.savefig(path + signal_type + '_' + str(current_slice) + '.png')
        plt.close(fig)


class Recorder:
    def __init__(self, slice_time):
        self.adxl345 = ADXL345()
        self.channels = []
        self.slice_time = slice_time
        self.current_slice = 1
        self.reference_standard_fft = []
        for i in range(number_of_channels):
            self.channels.append(Channel())

    def start_recording(self, duration, slice_nottime, record_reference_standard, save_graphs):
        if (record_reference_standard):
            self.record_reference_standard()
            print("Reference standard recorded")

        start_time = time.time()
        end_time = start_time + duration
        self.current_slice = 1

        while ((not slice_nottime and time.time() < end_time) or (slice_nottime and self.current_slice <= duration)):
            self.reset_channels()
            self.make_slice()
            if (save_graphs):
                self.save_slice_graphs()
            self.analyze_slice()

            print("Slice " + str(self.current_slice) + " taken")

            self.current_slice += 1

    def record_reference_standard(self):
        self.make_slice()
        self.reference_standard_fft = []
        for i in range(number_of_channels):
            self.reference_standard_fft.append(self.channels[i].channel_fft)

    def reset_channels(self):
        for i in range(number_of_channels):
            self.channels[i].reset()

    def get_channels_raw_data(self):
        channels_raw_data = self.adxl345.getAxes(True)  # False = m/s^2, True = g
        # TODO: add audio channel

        return channels_raw_data

    def make_slice(self):
        start_time = time.time()
        end_time = start_time + self.slice_time
        while (time.time() < end_time):
            channels_raw_data = self.get_channels_raw_data()

            for i in range(number_of_channels):
                self.channels[i].add_data(channels_raw_data[i])

        for i in range(number_of_channels):
            self.channels[i].subtract_average()
            self.channels[i].calculate_fft()

    def save_slice_graphs(self):
        self.channels[0].save_graphs(self.current_slice, "X")
        self.channels[1].save_graphs(self.current_slice, "Y")
        self.channels[2].save_graphs(self.current_slice, "Z")
        # TODO: save audio

    def analyze_slice(self):
        # TODO: analyze slice
        # TODO: notify user
        pass


def mainprog():
    print("Start")
    recorder = Recorder(slice_time)
    recorder.start_recording(duration, slice_not_time, record_reference_standard, save_graphs)


if __name__ == "__main__":
    mainprog()
