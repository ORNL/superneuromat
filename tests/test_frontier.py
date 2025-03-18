import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0,"../")

from src.superneuromat import NeuromorphicModel


class TestFrontier(unittest.TestCase):
	""" Tests the SNN simulation on the Frontier supercomputer
	"""

	def test_frontier(self):
		"""
		"""

		# print("[Python Test] Testing frontier")

		# snn = superneuromat.NeuromorphicModel(backend="frontier")

		# snn.simulate()

		print("This functionality is currently being built")



if __name__ == "__main__":
	unittest.main()