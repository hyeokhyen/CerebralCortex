# Copyright (c) 2016, MD2K Center of Excellence
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

import pytz

from cerebralcortex.data_processor.feature.ecg import ecg_feature_computation, lomb, heart_rate_power
from cerebralcortex.data_processor.signalprocessing.ecg import compute_rr_intervals
from cerebralcortex.data_processor.signalprocessing.window import window_sliding, window_iter
from cerebralcortex.kernel.datatypes.datapoint import DataPoint
from cerebralcortex.kernel.datatypes.datastream import DataStream

from pprint import pprint

import time

class TestECGFeatures(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		super(TestECGFeatures, cls).setUpClass()
		tz = pytz.timezone('US/Eastern')
		ecg = []
		ecg_sampling_frequency = 64.0
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/ecg.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				ecg.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))

		ecg_ds = DataStream(None, None)
		ecg_ds.data = ecg
		#print (ecg_ds.data)
		#print (len(ecg_ds.data))
		
		cls.ecg_ds = ecg_ds
		cls.ecg_sampling_frequency = ecg_sampling_frequency
		cls.rr_intervals = compute_rr_intervals(ecg_ds, ecg_sampling_frequency)
		#pprint (cls.rr_intervals.data)
	
	def test_compare_running_time(self):
		start_time_whole = time.time()
		rr_intervals = compute_rr_intervals(self.ecg_ds, self.ecg_sampling_frequency)
		elapse_time_whole = time.time() - start_time_whole
		print ('elapse_time_whole =', elapse_time_whole)
		
		window_size = 60
		window_offset = 1
		start_time_win = time.time()
		for key, data in window_iter(self.ecg_ds.data, window_size, window_offset):
			elapse_time_win = time.time() - start_time_win
			print (len(data), 'time =', '{:.2f}'.format(elapse_time_win), end='\r', flush=True)
			rr_intervals = compute_rr_intervals(DataStream(None, None, data=data), self.ecg_sampling_frequency)
		# ====> ALways whole!!!!!!

	def test_lomb(self):
		window_data = window_sliding(self.rr_intervals.data, window_size=120, window_offset=60)
		test_window = list(window_data.items())[0]
		result, frequency_range = lomb(test_window[1], low_frequency=0.01, high_frequency=0.7)
		self.assertAlmostEqual(result[0], 37.093948304468618, delta=0.01)
		self.assertEqual(frequency_range[-1], 0.7)

	def test_heart_rate_power(self):
		window_data = window_sliding(self.rr_intervals.data, window_size=120, window_offset=60)
		test_window = list(window_data.items())[0]
		power, frequency_range = lomb(test_window[1], low_frequency=0.01, high_frequency=0.7)
		hr_hf = heart_rate_power(power, frequency_range, 0.15, 0.4)
		self.assertAlmostEqual(hr_hf, 14.143026468160871, delta=0.01)

	def test_ecg_feature_computation(self):
		wSize = 60
		wOff = 60
		rr_variance, rr_vlf, rr_hf, rr_lf, rr_lf_hf, rr_mean, rr_median, rr_quartile, rr_80, rr_20, rr_heart_rate = \
			ecg_feature_computation(self.rr_intervals,
									window_size=wSize,
									window_offset=wOff)
		#print (rr_variance)

		# test all are DataStream
		self.assertIsInstance(rr_variance, DataStream)
		self.assertIsInstance(rr_vlf, DataStream)
		self.assertIsInstance(rr_hf, DataStream)
		self.assertIsInstance(rr_lf, DataStream)
		self.assertIsInstance(rr_lf_hf, DataStream)
		self.assertIsInstance(rr_mean, DataStream)
		self.assertIsInstance(rr_median, DataStream)
		self.assertIsInstance(rr_quartile, DataStream)
		self.assertIsInstance(rr_80, DataStream)
		self.assertIsInstance(rr_20, DataStream)
		self.assertIsInstance(rr_heart_rate, DataStream)

		self.assertAlmostEqual(rr_variance.data[0].sample, 0.1458641582638889, delta=0.01)
		self.assertAlmostEqual(rr_vlf.data[0].sample, 194.05492363605134, delta=0.01)
		self.assertAlmostEqual(rr_hf.data[0].sample, 14.143026468160871, delta=0.01)
		self.assertAlmostEqual(rr_lf.data[0].sample, 46.446940287356242, delta=0.01)
		self.assertAlmostEqual(rr_lf_hf.data[0].sample, 3.2840877722967385, delta=0.01)
		self.assertAlmostEqual(rr_mean.data[0].sample, 0.78799166666666665, delta=0.01)
		self.assertAlmostEqual(rr_median.data[0].sample, 0.61499999999999999, delta=0.01)
		self.assertAlmostEqual(rr_quartile.data[0].sample, 0.094874999999999987, delta=0.01)
		self.assertAlmostEqual(rr_80.data[0].sample, 1.0838000000000001, delta=0.01)
		self.assertAlmostEqual(rr_20.data[0].sample, 0.58499999999999996, delta=0.01)
		self.assertAlmostEqual(rr_heart_rate.data[0].sample, 97.56123355471891, delta=0.01)


if __name__ == '__main__':
	unittest.main()
