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
from typing import Tuple

from cerebralcortex.kernel.datatypes.datapoint import DataPoint
from cerebralcortex.kernel.datatypes.datastream import DataStream

import numpy as np
from pprint import pprint

def calculateTimeDelta(time1, time2):
	if time1 > time2:
		delta = time1 - time2
	else:
		delta = time2 - time1
	return delta

def rsaCalculateCycle(startTime, endTime, rrIntervals):

	list_rrInterval = []
	for i in range(len(rrIntervals)):
		rrDate = rrIntervals[i].start_time
		rrSample = rrIntervals[i].sample
		if rrDate > startTime and rrDate < endTime:
			list_rrInterval.append(rrSample)

	if len(list_rrInterval) > 0:
		max_rr = np.max(list_rrInterval)
		min_rr = np.min(list_rrInterval)
		rsa = max_rr - min_rr
	else:
		rsa = -1
	return rsa

def getStats(data_window, list_qDev, list_mean, list_median, list_80):
	# data array
	data = np.array([i.sample for i in data_window])

	# Quantile deviation
	value_qDev = (0.5 * (np.percentile(data, 75) - np.percentile(data, 25)))
	list_qDev.append(DataPoint.from_tuple(start_time=starttime, end_time=endtime, sample = value_qDev))
	# Mean
	value_mean = np.mean(data)
	list_mean.append(DataPoint.from_tuple(start_time=starttime, end_time=endtime, sample = value_mean))
	# Median
	value_median = np.median(data)
	list_median.append(DataPoint.from_tuple(start_time=starttime, end_time=endtime, sample = value_median))
	# 80 Percentile
	value_80 = np.percentile(data, 80)
	list_80.append(DataPoint.from_tuple(start_time=starttime, end_time=endtime, sample = value_80))
	return list_qDev, list_mean, list_median, list_80

def getStats_window(data):
	# Quantile deviation
	value_qDev = 0.5 * (np.percentile(data, 75) - np.percentile(data, 25))
	# Mean
	value_mean = np.mean(data)
	# Median
	value_median = np.median(data)
	# 80 Percentile
	value_80 = np.percentile(data, 80)

	return value_qDev, value_mean, value_median, value_80


def list2datastream(data_qDev, data_mean, data_median, data_80, rr_intervals_datastream):
	# Quantile deviation
	data_qDev_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_qDev_datastream.data = data_qDev
	# Mean
	data_mean_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_mean_datastream.data = data_mean
	# Median
	data_median_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_median_datastream.data = data_median
	# 80 percentile
	data_80_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_80_datastream.data = data_80
	return data_qDev_datastream, data_mean_datastream, \
		   data_median_datastream, data_80_datastream

def value2datastream(data_qDev, data_mean, data_median, data_80, rr_intervals_datastream):
	# Quantile deviation
	data_qDev_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_qDev_datastream.data = data_qDev
	# Mean
	data_mean_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_mean_datastream.data = data_mean
	# Median
	data_median_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_median_datastream.data = data_median
	# 80 percentile
	data_80_datastream = DataStream.from_datastream([rr_intervals_datastream])
	data_80_datastream.data = data_80
	return data_qDev_datastream, data_mean_datastream, \
		   data_median_datastream, data_80_datastream


def rip_feature_computation(peaks_datastream: DataStream,
							valleys_datastream: DataStream,
							rr_intervals_datastream: DataStream,
							window_size:float,
							window_offset:float) -> Tuple[DataStream]:
	"""
	Respiration Feature Implementation. The respiration feature values are
	derived from the following paper:
	'puffMarker: a multi-sensor approach for pinpointing the timing of first lapse in smoking cessation'


	Removed due to lack of current use in the implementation
	roc_max = []  # 8. ROC_MAX = max(sample[j]-sample[j-1])
	roc_min = []  # 9. ROC_MIN = min(sample[j]-sample[j-1])


	:param peaks_datastream: DataStream
	:param valleys_datastream: DataStream
	:return: RIP Feature DataStreams
	"""

	# TODO: This needs fixed to prevent crashing the execution pipeline
	if peaks_datastream is None or valleys_datastream is None:
		return None

	# TODO: This needs fixed to prevent crashing the execution pipeline
	if len(peaks_datastream.data) == 0 or len(valleys_datastream.data) == 0:
		return None

	inspiration_duration = []  # 1 Inhalation duration
	expiration_duration = []  # 2 Exhalation duration
	respiration_duration = []  # 3 Respiration duration
	inspiration_expiration_ratio = []  # 4 Inhalation and Exhalation ratio
	stretch = []  # 5 Stretch
	upper_stretch = []  # 6. Upper portion of the stretch calculation
	lower_stretch = []  # 7. Lower portion of the stretch calculation
	delta_previous_inspiration_duration = []  # 10. BD_INSP = INSP(i)-INSP(i-1)
	delta_previous_expiration_duration = []  # 11. BD_EXPR = EXPR(i)-EXPR(i-1)
	delta_previous_respiration_duration = []  # 12. BD_RESP = RESP(i)-RESP(i-1)
	delta_previous_stretch_duration = []  # 14. BD_Stretch= Stretch(i)-Stretch(i-1)
	delta_next_inspiration_duration = []  # 19. FD_INSP = INSP(i)-INSP(i+1)
	delta_next_expiration_duration = []  # 20. FD_EXPR = EXPR(i)-EXPR(i+1)
	delta_next_respiration_duration = []  # 21. FD_RESP = RESP(i)-RESP(i+1)
	delta_next_stretch_duration = []  # 23. FD_Stretch= Stretch(i)-Stretch(i+1)
	neighbor_ratio_expiration_duration = []  # 29. D5_EXPR(i) = EXPR(i) / avg(EXPR(i-2)...EXPR(i+2))
	neighbor_ratio_stretch_duration = []  # 32. D5_Stretch = Stretch(i) / avg(Stretch(i-2)...Stretch(i+2))

	#----------------------------------------------------------------------
	rsa = [] # RSA
	rrIntervals = rr_intervals_datastream.data
	#----------------------------------------------------------------------

	valleys = valleys_datastream.data
	peaks = peaks_datastream.data[:-1]

	for i, peak in enumerate(peaks):
		valley_start_time = valleys[i].start_time

		delta = peak.start_time - valleys[i].start_time
		inspiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta.total_seconds()))

		delta = valleys[i + 1].start_time - peak.start_time
		expiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta.total_seconds()))

		delta = valleys[i + 1].start_time - valley_start_time
		respiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta.total_seconds()))

		ratio = (peak.start_time - valley_start_time) / (valleys[i + 1].start_time - peak.start_time)
		inspiration_expiration_ratio.append(DataPoint.from_tuple(start_time=valley_start_time, sample=ratio))

		value = peak.sample - valleys[i + 1].sample
		stretch.append(DataPoint.from_tuple(start_time=valley_start_time, sample=value))

		#----------------------------------------------------------------------
		# RSA
		value = rsaCalculateCycle(valley_start_time, valleys[i + 1].start_time, rrIntervals)
		if value != -1:
			rsa.append(Data.Point.from_tuple(start_time=valley_start_time, sample=value))
		#----------------------------------------------------------------------
		# upper_stretch.append(DataPoint.from_tuple(start_time=valley_start_time, sample=(peak.sample - valleys[i + 1][1]) / 2))  # TODO: Fix this by adding a tracking moving average and compute upper stretch from this to the peak and lower from this to the valley
		# lower_stretch.append(DataPoint.from_tuple(start_time=valley_start_time, sample=(-peak.sample + valleys[i + 1][1]) / 2))  # TODO: Fix this by adding a tracking moving average and compute upper stretch from this to the peak and lower from this to the valley

	for i in range(len(inspiration_duration)):
		valley_start_time = valleys[i].start_time
		if i == 0:  # Edge case
			delta_previous_inspiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
			delta_previous_expiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
			delta_previous_respiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
			delta_previous_stretch_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
		else:
			delta = inspiration_duration[i].sample - inspiration_duration[i - 1].sample
			delta_previous_inspiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

			delta = expiration_duration[i].sample - expiration_duration[i - 1].sample
			delta_previous_expiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

			delta = respiration_duration[i].sample - respiration_duration[i - 1].sample
			delta_previous_respiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

			delta = stretch[i].sample - stretch[i - 1].sample
			delta_previous_stretch_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

		if i == len(inspiration_duration) - 1:
			delta_next_inspiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
			delta_next_expiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
			delta_next_respiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
			delta_next_stretch_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=0.0))
		else:
			delta = inspiration_duration[i].sample - inspiration_duration[i + 1].sample
			delta_next_inspiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

			delta = expiration_duration[i].sample - expiration_duration[i + 1].sample
			delta_next_expiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

			delta = respiration_duration[i].sample - respiration_duration[i + 1].sample
			delta_next_respiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

			delta = stretch[i].sample - stretch[i + 1].sample
			delta_next_stretch_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=delta))

		stretch_average = 0
		expiration_average = 0
		count = 0.0
		for j in [-2, -1, 1, 2]:
			if i + j < 0 or i + j >= len(inspiration_duration):
				continue
			stretch_average += stretch[i + j].sample
			expiration_average += expiration_duration[i + j].sample
			count += 1

		stretch_average /= count
		expiration_average /= count

		ratio = stretch[i].sample / stretch_average
		neighbor_ratio_stretch_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=ratio))

		ratio = expiration_duration[i].sample / expiration_average
		neighbor_ratio_expiration_duration.append(DataPoint.from_tuple(start_time=valley_start_time, sample=ratio))

	# Begin assembling datastream for output
	inspiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	inspiration_duration_datastream.data = inspiration_duration

	expiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	expiration_duration_datastream.data = expiration_duration

	respiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	respiration_duration_datastream.data = respiration_duration

	inspiration_expiration_ratio_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	inspiration_expiration_ratio_datastream.data = inspiration_expiration_ratio

	stretch_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	stretch_datastream.data = stretch

	#----------------------------------------------------------------------
	rsa_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	rsa_datastream.data = rsa
	#----------------------------------------------------------------------

	upper_stretch_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	upper_stretch_datastream.data = upper_stretch

	lower_stretch_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	lower_stretch_datastream.data = lower_stretch

	delta_previous_inspiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_previous_inspiration_duration_datastream.data = delta_previous_inspiration_duration

	delta_previous_expiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_previous_expiration_duration_datastream.data = delta_previous_expiration_duration

	delta_previous_respiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_previous_respiration_duration_datastream.data = delta_previous_respiration_duration

	delta_previous_stretch_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_previous_stretch_duration_datastream.data = delta_previous_stretch_duration

	delta_next_inspiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_next_inspiration_duration_datastream.data = delta_next_inspiration_duration

	delta_next_expiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_next_expiration_duration_datastream.data = delta_next_expiration_duration

	delta_next_respiration_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_next_respiration_duration_datastream.data = delta_next_respiration_duration

	delta_next_stretch_duration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	delta_next_stretch_duration_datastream.data = delta_next_stretch_duration

	neighbor_ratio_expiration_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	neighbor_ratio_expiration_datastream.data = neighbor_ratio_expiration_duration

	neighbor_ratio_stretch_datastream = DataStream.from_datastream([peaks_datastream, valleys_datastream])
	neighbor_ratio_stretch_datastream.data = neighbor_ratio_stretch_duration


	#----------------------------------------------------------------------
	# Aggregate minute level feature
	# Breath-rate
	valleys_window = window_sliding(valleys, window_size, window_offset)
	peaks_window = window_sliding(valleys, window_size, window_offset)
	insp_window = window_sliding(inspiration_duration, window_size, window_offset)
	exp_window = window_sliding(expiration_duration, window_size, window_offset)
	resp_window = window_sliding(respiration_duration, window_size, window_offset)
	inspExp_window = window_sliding(inspiration_expiration_ratio, window_size, window_offset)
	stretch_window = window_sliding(stretch, window_size, window_offset)
	rsa_window = window_sliding(rsa, window_size, window_offset)

	breath_rate = []
	insp_min_vol = []
	insp_qDev = [];    insp_mean = [];    insp_median = [];    insp_80 = []
	exp_qDev = [];     exp_mean = [];     exp_median = [];     exp_80 = []
	resp_qDev = [];    resp_mean = [];    resp_median = [];    resp_80 = []
	inspExp_qDev = []; inspExp_mean = []; inspExp_median = []; inspExp_80 = []
	stretch_qDev = []; stretch_mean = []; stretch_median = []; stretch_80 = []
	rsa_qDev = [];     rsa_mean = [];     rsa_median = [];     rsa_80 = []

	for key, value in valleys_window.items():
		starttime, endtime = key

		# breath_rate
		breath_rate.append(DataPoint.from_tuple(start_time = starttime,
												end_time = endtime,
												sample = len(valleys_window[key])))
		# inspiration minute volume
		value = 0.;
		for i in range(len(valleys_window[key])):
			peak_value = peaks_window[key][i].sample
			peak_time = peaks_window[key][i].start_time.timestamp()
			valley_value = valleys_window[key][i].sample
			valley_time = valleys_window[key][i].start_time.timestamp()
			if peak_time > valley_time:
				value += (peak_time - valley_time) * (peak_value - valley_value) / 2
		insp_min_vol.append(DataPoint.from_tuple(start_time = starttime,
												 end_time = endtime,
												 sample = value))
		# inspiration duration
		insp_qDev, insp_mean, insp_median, insp_80 = getStats(
					insp_window[key], insp_qDev, insp_mean, insp_median, insp_80)
		# Expiration duration
		exp_qDev, exp_mean, exp_median, exp_80 = getStats(
					exp_window[key], exp_qDev, exp_mean, exp_median, exp_80)
		# Respiration duration
		resp_qDev, resp_mean, resp_median, resp_80 = getStats(
					resp_window[key], resp_qDev, resp_mean, resp_median, resp_80)
		# Inspiration Expiration duration ratio
		inspExp_qDev, inspExp_mean, inspExp_median, inspExp_80 = getStats(
					inspExp_window[key], inspExp_qDev, inspExp_mean, inspExp_median, inspExp_80)
		# Stretch
		stretch_qDev, stretch_mean, stretch_median, stretch_80 = getStats(
					stretch_window[key], stretch_qDev, stretch_mean, stretch_median, stretch_80)
		# RSA
		rsa_qDev, rsa_mean, rsa_median, rsa_80 = getStats(
					rsa_window[key], rsa_qDev, rsa_mean, rsa_median, rsa_80)

	# To datastream struct
	# breath_rate
	breath_rate_datastream = DataStream.from_datastream([rr_intervals_datastream])
	breath_rate_datastream.data = breath_rate
	# inspiration minute volume
	insp_min_vol_datastream = DataStream.from_datastream([rr_intervals_datastream])
	insp_min_vol_datastream.data = insp_min_vol
	# Inspiration duration
	insp_qDev_datastream, insp_mean_datastream, insp_median_datastream, insp_80_datastream = \
			list2datastream(insp_qDev, insp_mean, insp_median, insp_80, rr_intervals_datastream)
	# Expiration duration
	exp_qDev_datastream, exp_mean_datastream, exp_median_datastream, exp_80_datastream = \
			list2datastream(exp_qDev, exp_mean, exp_median, exp_80, rr_intervals_datastream)
	# Respiration Duration
	resp_qDev_datastream, resp_mean_datastream, resp_median_datastream, resp_80_datastream = \
			list2datastream(resp_qDev, resp_mean, resp_median, resp_80, rr_intervals_datastream)
	# Inspiration Expiration duration ratio
	inspExp_qDev_datastream, inspExp_mean_datastream, inspExp_median_datastream, inspExp_80_datastream = \
			list2datastream(inspExp_qDev, inspExp_mean, inspExp_median, inspExp_80, rr_intervals_datastream)
	# Stretch
	stretch_qDev_datastream, stretch_mean_datastream, stretch_median_datastream, stretch_80_datastream = \
			list2datastream(stretch_qDev, stretch_mean, stretch_median, stretch_80, rr_intervals_datastream)
	# RSA
	rsa_qDev_datastream, rsa_mean_datastream, rsa_median_datastream, rsa_80_datastream = \
			list2datastream(rsa_qDev, rsa_mean, rsa_median, rsa_80, rr_intervals_datastream)
	#----------------------------------------------------------------------

	return breath_rate_datastream, insp_min_vol_datastream, \
		   insp_qDev_datastream,    insp_mean_datastream,    insp_median_datastream,    insp_80_datastream, \
		   exp_qDev_datastream,     exp_mean_datastream,     exp_median_datastream,     exp_80_datastream, \
		   resp_qDev_datastream,    resp_mean_datastream,    resp_median_datastream,    resp_80_datastream, \
		   inspExp_qDev_datastream, inspExp_mean_datastream, inspExp_median_datastream, inspExp_80_datastream, \
		   stretch_qDev_datastream, stretch_mean_datastream, stretch_median_datastream, stretch_80_datastream, \
		   rsa_qDev_datastream,     rsa_mean_datastream,     rsa_median_datastream,     rsa_80_datastream

	'''
	return inspiration_duration_datastream, \
		   expiration_duration_datastream, \
		   respiration_duration_datastream, \
		   inspiration_expiration_ratio_datastream, \
		   stretch_datastream, \
		   upper_stretch_datastream, \
		   lower_stretch_datastream, \
		   delta_previous_inspiration_duration_datastream, \
		   delta_previous_expiration_duration_datastream, \
		   delta_previous_respiration_duration_datastream, \
		   delta_previous_stretch_duration_datastream, \
		   delta_next_inspiration_duration_datastream, \
		   delta_next_expiration_duration_datastream, \
		   delta_next_respiration_duration_datastream, \
		   delta_next_stretch_duration_datastream, \
		   neighbor_ratio_expiration_datastream, \
		   neighbor_ratio_stretch_datastream
	'''

def rip_window_feature_computation(peaks_datastream: DataStream,
							valleys_datastream: DataStream,
							rr_intervals_datastream: DataStream):
	"""
	Respiration Feature Implementation. The respiration feature values are
	derived from the following paper:
	'puffMarker: a multi-sensor approach for pinpointing the timing of first lapse in smoking cessation'


	Removed due to lack of current use in the implementation
	roc_max = []  # 8. ROC_MAX = max(sample[j]-sample[j-1])
	roc_min = []  # 9. ROC_MIN = min(sample[j]-sample[j-1])


	:param peaks_datastream: DataStream
	:param valleys_datastream: DataStream
	:return: RIP Feature DataStreams
	"""

	# TODO: This needs fixed to prevent crashing the execution pipeline
	if peaks_datastream is None or valleys_datastream is None:
		return None

	# TODO: This needs fixed to prevent crashing the execution pipeline
	if len(peaks_datastream.data) == 0 or len(valleys_datastream.data) == 0:
		return None

	inspiration_duration = []  # 1 Inhalation duration
	expiration_duration = []  # 2 Exhalation duration
	respiration_duration = []  # 3 Respiration duration
	inspiration_expiration_ratio = []  # 4 Inhalation and Exhalation ratio
	stretch = []  # 5 Stretch

	#----------------------------------------------------------------------
	rsa = [] # RSA
	rrIntervals = rr_intervals_datastream.data
	#----------------------------------------------------------------------

	valleys = valleys_datastream.data
	peaks = peaks_datastream.data[:len(valleys)-1]
	#peaks = peaks_datastream.data[:-1]	
	#print (len(valleys), len(peaks))

	for i, peak in enumerate(peaks):
		valley_start_time = valleys[i].start_time

		delta = calculateTimeDelta(peak.start_time, valleys[i].start_time)
		inspiration_duration.append(delta.total_seconds())

		delta = calculateTimeDelta(valleys[i + 1].start_time, peak.start_time)
		expiration_duration.append(delta.total_seconds())

		delta = calculateTimeDelta(valleys[i + 1].start_time, valley_start_time)
		respiration_duration.append(delta.total_seconds())

		delta_insp = calculateTimeDelta(peak.start_time, valley_start_time)
		delta_exp = calculateTimeDelta(valleys[i + 1].start_time, peak.start_time)
		ratio = delta_insp / delta_exp
		inspiration_expiration_ratio.append(ratio)
		
		value = np.absolute(peak.sample - valleys[i + 1].sample) 
		stretch.append(value)

		#----------------------------------------------------------------------
		# RSA
		value = rsaCalculateCycle(valley_start_time, valleys[i + 1].start_time, rrIntervals)
		if value != -1:
			rsa.append(value)
		#----------------------------------------------------------------------

	#print (len(rsa))
	if len(rsa) == 0:
		result = []
		return result
	#----------------------------------------------------------------------
	# Aggregate minute level feature

	# breath_rate
	breath_rate = len(valleys)

	# inspiration minute volume
	valleys = valleys_datastream.data
	peaks = peaks_datastream.data
	num_data = np.min([len(valleys), len(peaks)])
	value = 0.;
	for i in range(num_data):
		peak_value = peaks[i].sample
		peak_time = peaks[i].start_time.timestamp()
		valley_value = valleys[i].sample
		valley_time = valleys[i].start_time.timestamp()
		delta_time = calculateTimeDelta(peak_time, valley_time)
		delta_value = np.absolute(peak_value - valley_value)
		value += delta_time * delta_value / 2
	insp_min_vol = value
	# inspiration duration
	insp_qDev, insp_mean, insp_median, insp_80 = getStats_window(inspiration_duration)
	# Expiration duration
	exp_qDev, exp_mean, exp_median, exp_80 = getStats_window(expiration_duration)
	# Respiration duration
	resp_qDev, resp_mean, resp_median, resp_80 = getStats_window(respiration_duration)
	# Inspiration Expiration duration ratio
	inspExp_qDev, inspExp_mean, inspExp_median, inspExp_80 = getStats_window(inspiration_expiration_ratio)
	# Stretch
	stretch_qDev, stretch_mean, stretch_median, stretch_80 = getStats_window(stretch)
	# RSA
	rsa_qDev, rsa_mean, rsa_median, rsa_80 = getStats_window(rsa)
	#----------------------------------------------------------------------

	result = [breath_rate, insp_min_vol,
		   insp_qDev,    insp_mean,    insp_median,    insp_80,
		   exp_qDev,     exp_mean,     exp_median,     exp_80,
		   resp_qDev,    resp_mean,    resp_median,    resp_80,
		   inspExp_qDev, inspExp_mean, inspExp_median, inspExp_80,
		   stretch_qDev, stretch_mean, stretch_median, stretch_80,
		   rsa_qDev,     rsa_mean,     rsa_median,     rsa_80]
		   
	return result