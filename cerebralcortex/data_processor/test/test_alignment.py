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

from cerebralcortex.data_processor.signalprocessing.alignment import interpolate_gaps, timestamp_correct
from cerebralcortex.kernel.datatypes.datapoint import DataPoint
from cerebralcortex.kernel.datatypes.datastream import DataStream

from cerebralcortex.data_processor.signalprocessing.window import window_iter

import time

class TestAlignment(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		super(TestAlignment, cls).setUpClass()
		tz = pytz.timezone('US/Eastern')

		cls.sample_rate = 64.0 / 6.0

		cls.accelx = []
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/accelx.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				cls.accelx.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
		cls.accelx = DataStream(None, None, data=cls.accelx)

		cls.accely = []
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/accely.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				cls.accely.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
		cls.accely = DataStream(None, None, data=cls.accely)

		cls.accelz = []
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/accelz.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				cls.accelz.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
		cls.accelz = DataStream(None, None, data=cls.accelz)

	def test_interpolate_gaps(self):
		result = interpolate_gaps(self.accelx.data, self.sample_rate)

		self.assertEqual(len(self.accelx.data), 63598)
		self.assertEqual(len(result), 65964)

	def test_timestamp_correct(self):
		result = timestamp_correct(self.accelx, sampling_frequency=self.sample_rate)

		self.assertEqual(len(self.accelx.data), 63598)
		self.assertEqual(len(result.data), 70010)
	
	def test_compare_running_time(self):
		start_time_whole = time.time()
		result = timestamp_correct(self.accelx, sampling_frequency=self.sample_rate)
		elapse_time_whole = time.time() - start_time_whole
		print ('elapse_time_whole =', elapse_time_whole)
		
		window_size = 60
		window_offset = 1
		start_time_win = time.time()
		for key, data in window_iter(self.accelx.data, window_size, window_offset):
			elapse_time_win = time.time() - start_time_win
			print (len(data), data[-1].start_time, 'time =', '{:.2f}'.format(elapse_time_win),  end='\r', flush=True)
			result = timestamp_correct(DataStream(None, None, data=data), sampling_frequency=self.sample_rate)
			
		# ====> ALways whole!!!!!!

if __name__ == '__main__':
	unittest.main()
