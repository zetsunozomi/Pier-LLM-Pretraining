Apr 12:

Changed the implementation of create subgroup： no longer uses mydistributed.py, but use parallel_state to create the subgroup.

Problem1: The validation loss seems not divided into 2 groups correctly.

Problem2: The nccl error makes the program can't run for the second time.