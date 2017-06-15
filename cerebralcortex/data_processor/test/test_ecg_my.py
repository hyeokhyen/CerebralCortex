# Copyright (c) 2017, MD2K Center of Excellence
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime
import gzip
import os
import unittest

import numpy as np
import pytz

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from cerebralcortex.data_processor.signalprocessing.alignment import timestamp_correct
from cerebralcortex.data_processor.signalprocessing.ecg import rr_interval_update, compute_moving_window_int, \
	check_peak, compute_r_peaks, remove_close_peaks, confirm_peaks, compute_rr_intervals
from cerebralcortex.kernel.datatypes.datapoint import DataPoint
from cerebralcortex.kernel.datatypes.datastream import DataStream


class TestRPeakDetect(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		super(TestRPeakDetect, cls).setUpClass()
		tz = pytz.timezone('US/Eastern')
		cls.ecg = []
		cls._fs = 64.0
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/ecg.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				cls.ecg.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
		cls.ecg_datastream = DataStream(None, None)
		cls.ecg_datastream.data = cls.ecg
		print (len(cls.ecg_datastream.data))
		
		start_time = cls.ecg_datastream.data[0].start_time
		end_time = cls.ecg_datastream.data[-1].start_time
		print (start_time)
		print (end_time)
		print (np.datetime64(end_time).astype(np.int64) - np.datetime64(start_time).astype(np.int64))
		
	def test_compute_rr_intervals(self):
		ecg_rrInterval, ecg_rpeak = compute_rr_intervals(self.ecg_datastream,fs=self._fs)
		
		#plot
		fig, ax = plt.subplots(figsize=(400, 10))
		ecg_time = [i.start_time for i in self.ecg_datastream.data]
		ecg_sample = [i.sample for i in self.ecg_datastream.data]
		rpeak_time = [i.start_time for i in ecg_rpeak.data]
		rpeak_sample = [i.sample for i in ecg_rpeak.data]
		
		ax.plot(ecg_time, ecg_sample, 'ro--', markersize=3, linewidth=1)
		ax.plot(rpeak_time, rpeak_sample, 'bo', markersize=5)
		
		path_save = '/home/hyeok/research/md2k/data/minnesota-analysis/test/rpeaks.png'
		fig.savefig(path_save)


if __name__ == '__main__':
	unittest.main()
