"""
Any thing that iterates Text, FloatTensor of waves and ids is a dataloader.
"""

from .mix_vctk_ljs import BinnedBatchLoader as DataLoader
