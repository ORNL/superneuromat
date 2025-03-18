import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0,"../")

# from superneuromat.neuromorphicmodel import NeuromorphicModel
from superneuromat import NeuromorphicModel
# import superneuromat


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