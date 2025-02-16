# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class EvalMetrics:
	"""
	Evaluation metrics.
	"""
	acc: float = 0.0
	precision: float = 0.0
	recall: float = 0.0
	f1: float = 0.0

	def __str__(self):
		return f"Accuracy: {self.acc:.4f}, Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}"
