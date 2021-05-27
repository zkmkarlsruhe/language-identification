"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

class Range(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end

	def __eq__(self, other):
		return self.start <= other <= self.end

	def __contains__(self, item):
		return self.__eq__(item)

	def __iter__(self):
		yield self

	def __str__(self):
		return '{0} - {1}'.format(self.start, self.end)
