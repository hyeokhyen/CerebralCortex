import datetime
import gzip
import os
import sys
import unittest

import pytz
import numpy as np
import pandas as pd

from cerebralcortex.data_processor.feature.rip_rsa import rip_window_feature_computation
from cerebralcortex.data_processor.signalprocessing.rip import compute_peak_valley

from cerebralcortex.data_processor.feature.ecg import ecg_feature_computation, lomb, heart_rate_power
from cerebralcortex.data_processor.signalprocessing.ecg import compute_rr_intervals

from cerebralcortex.kernel.datatypes.datapoint import DataPoint
from cerebralcortex.kernel.datatypes.datastream import DataStream
from cerebralcortex.data_processor.signalprocessing.window import window_sliding

from pprint import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from collections import OrderedDict

import warnings
warnings.simplefilter("error")

class TestAllFeatures(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		super(TestAllFeatures, cls).setUpClass()
		tz = pytz.timezone('US/Eastern')
		# Load RIP
		rip = []
		rip_sampling_frequency = 64.0 / 3
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/rip.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				rip.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
		# Load ECG
		ecg = []
		ecg_sampling_frequency = 64.0
		with gzip.open(os.path.join(os.path.dirname(__file__), 'res/ecg.csv.gz'), 'rt') as f:
			for l in f:
				values = list(map(int, l.split(',')))
				ecg.append(
					DataPoint.from_tuple(datetime.datetime.fromtimestamp(values[0] / 1000000.0, tz=tz), values[1]))
		# Dataframe
		ecg_date = np.array([np.nan]*len(ecg)).astype(datetime.datetime)
		ecg_sample = np.array([np.nan]*len(ecg))
		rip_date = np.array([np.nan]*len(rip)).astype(datetime.datetime)
		rip_sample = np.array([np.nan]*len(rip))
		for i in range(len(ecg)):
			ecg_date[i] = ecg[i].start_time
			ecg_sample[i] = ecg[i].sample
		for i in range(len(rip)):
			rip_date[i] = rip[i].start_time
			rip_sample[i] = rip[i].sample

		df_ecg = pd.DataFrame(index=np.arange(len(ecg)),
							  data = ecg_sample,
							  columns= ['sample'])
		df_ecg['Date'] = ecg_date
		df_rip = pd.DataFrame(index=np.arange(len(rip)),
							  data = rip_sample,
							  columns =['sample'])
		df_rip['Date'] = rip_date

		print (df_rip.head())
		print (df_rip.tail())
		print (df_ecg.head())
		print (df_ecg.tail())

		cls.df_ecg = df_ecg
		cls.df_rip = df_rip
		
		cls.sampling_frequency = {}
		cls.sampling_frequency['ecg'] = 64.0
		cls.sampling_frequency['rip'] = 64.0 / 3
		

	def test_features_window(self):
		# path include stress intervention
		path_stressInter = '/home/hyeok/research/md2k/code/stressIntervension'
		sys.path.append(path_stressInter)
		from util.explore import features_window

		window_size = 60
		window_offset = 30

		features = OrderedDict()
		for key, data in features_window(self, self.df_ecg, self.df_rip, window_size, window_offset):
			features[key] = data
		
		items = list(features.items())
		pprint (items[0])
		pprint (items[1])
		pprint (items[-2])
		pprint (items[-1])
		

if __name__ == '__main__':
	unittest.main()
