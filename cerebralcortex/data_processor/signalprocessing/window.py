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

import math
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List

import pytz
from pprint import pprint
import numpy as np

from cerebralcortex.kernel.datatypes.datapoint import DataPoint

def epoch_align(ts: datetime,
				offset: float,
				after: bool = False,
				time_zone: pytz = pytz.timezone('US/Central'),
				time_base: int = 1e6) -> datetime:
	"""
	Epoch timestamp alignment based on offset

	:param time_zone: Specifiy the timezone of the timestamps, default US/Central
	:param ts: datatime object representing the timestamp to start with
	:param offset: seconds as a float
	:param after: Flag designating if the result should be after ts
	:param time_base: specifies the precision with which the time base should be manipulated (1e6 -> microseconds)
	:return: aligned datetime object
	"""
	new_timestamp = math.floor(ts.timestamp() * time_base / (offset * time_base)) * offset * time_base

	if after:
		new_timestamp += offset * time_base

	result = datetime.fromtimestamp(new_timestamp / time_base, time_zone)
	return result


def window(data: List[DataPoint],
		   window_size: float) -> OrderedDict:
	"""
	Special case of a sliding window with no overlaps
	:param data:
	:param window_size:
	:return:
	"""
	return window_sliding(data, window_size=window_size, window_offset=window_size)


def window_sliding(data: List[DataPoint],
				   window_size: float,
				   window_offset: float) -> OrderedDict:
	"""
	Sliding Window Implementation

	:param data: list
	:param window_size: float
	:param window_offset: float
	:return: OrderedDict representing [(st,et),[dp,dp,dp,dp...],
									   (st,et),[dp,dp,dp,dp...],
										...]
	"""
	if data is None:
		raise TypeError('data is not List[DataPoint]')

	if len(data) == 0:
		raise ValueError('The length of data is zero')

	windowed_datastream = OrderedDict()

	for key, data in window_iter(data, window_size, window_offset):
		windowed_datastream[key] = data

	return windowed_datastream

def window_iter(iterable: List[DataPoint],
				window_size: float,
				window_offset: float):
	"""
	Window iteration function that support various common implementations
	:param iterable:
	:param window_size:
	:param window_offset:
	"""

	start_time = epoch_align(iterable[0].start_time, window_offset)
	final_time = iterable[-1].start_time
	win_size = timedelta(seconds=window_size)
	window_offset_delta = timedelta(seconds=window_offset)
 	
	while start_time < final_time:
		end_time = start_time + win_size
		key = (start_time, end_time)
		
		data = []
		num_data = len(iterable)
		for i in range(num_data):
			if iterable[i].start_time > end_time:
				iterable = iterable[i:]
				break
			elif iterable[i].start_time > start_time:
				data.append(iterable[i])

		if len(data) == 0:
			num_data = len(iterable)
			for i in range(num_data):
				if iterable[i].start_time > end_time:
					start_time = iterable[i].start_time
					start_time = epoch_align(start_time, window_offset)
					iterable = iterable[i:]
					break
		else:
			# slide window
			start_time = start_time + window_offset_delta
			'''
			# check data length in window
			data_start_time = np.datetime64(data[0].start_time).astype(np.int64)/10**6
			data_end_time = np.datetime64(data[-1].start_time).astype(np.int64)/10**6
			data_timedelta = data_end_time - data_start_time
			if data_timedelta > window_offset/2:
				yield key, data
			'''
			yield key, data

def window_sliding_multi(data: List[DataPoint],
				   window_size: float,
				   window_offset: float) -> OrderedDict:
	"""
	Sliding Window Implementation

	:param data: list
	:param window_size: float
	:param window_offset: float
	:return: OrderedDict representing [(st,et),[dp,dp,dp,dp...],
									   (st,et),[dp,dp,dp,dp...],
										...]
	"""
	if data is None:
		raise TypeError('data is not List[DataPoint]')

	windowed_datastream = OrderedDict()
	for key, data in window_iter_multi(data, window_size, window_offset):
		windowed_datastream[key] = data

	return windowed_datastream

def window_iter_multi(iterable_dict,
				window_size: float,
				window_offset: float):
	"""
	Window iteration function that support various common implementations
	:param iterable:
	:param window_size:
	:param window_offset:
	"""

	peak = iterable_dict['peak']
	valley = iterable_dict['valley']
	rr_intervals = iterable_dict['rr_intervals']

	start_time = np.max([peak[0].start_time, valley[0].start_time, rr_intervals[0].start_time])
	final_time = np.min([peak[-1].start_time, valley[-1].start_time, rr_intervals[-1].start_time])

	start_time = epoch_align(start_time, window_offset)
	win_size = timedelta(seconds=window_size)
	window_offset_delta = timedelta(seconds=window_offset)

	while start_time < final_time:
		end_time = start_time + win_size
		key = (start_time, end_time)

		data = {}
		# valley -> peak
		data['valley'] = []
		data['peak'] = []
		num_data = len(valley)
		for i in range(num_data):
			if valley[i].start_time > end_time:
				valley = valley[i:]
				peak = peak[i:]
				break
			elif valley[i].start_time > start_time:
				data['valley'].append(valley[i])
				data['peak'].append(peak[i])
		data['rr_intervals'] = []
		num_data = len(rr_intervals)
		for i in range(num_data):
			if rr_intervals[i].start_time > end_time:
				rr_intervals = rr_intervals[i:]
				break
			elif rr_intervals[i].start_time > start_time:
				data['rr_intervals'].append(rr_intervals[i])
		
		if len(data['peak']) == 0 or len(data['valley']) == 0 or len(data['rr_intervals']) == 0:
			# valley -> peak
			num_data = len(valley)
			for i in range(num_data):
				if valley[i].start_time > start_time:
					start_time = valley[i].start_time
					valley = valley[i:]
					peak = peak[i:]
					break
			num_data = len(rr_intervals)
			for i in range(num_data):
				if rr_intervals[i].start_time > start_time:
					start_time = rr_intervals[i].start_time
					rr_intervals = rr_intervals[i:]
					break
			start_time = epoch_align(start_time, window_offset)
		else:
			# slide window
			start_time = start_time + window_offset_delta
			
			# check data length in window
			data_start_time = np.max([data['peak'][0].start_time, data['valley'][0].start_time, data['rr_intervals'][0].start_time])
			data_start_time = np.datetime64(data_start_time).astype(np.int64)/10**6
			data_end_time = np.min([data['peak'][-1].start_time, data['valley'][-1].start_time, data['rr_intervals'][-1].start_time])
			data_end_time = np.datetime64(data_end_time).astype(np.int64)/10**6
			data_timedelta = data_end_time - data_start_time
			if data_timedelta > window_offset/2:
				yield key, data
