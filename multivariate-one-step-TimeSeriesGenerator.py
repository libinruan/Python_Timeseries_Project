# multivariate one step problem
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
print(dataset)
# define generator
n_input = 2
generator = TimeseriesGenerator(dataset, dataset, length=n_input, batch_size=1)
# number of samples
print('\n# Samples: %d' % len(generator))
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))

# Result:
# [[ 10  15]  # = t1
#  [ 20  25]  # = t2
#  [ 30  35]  # = t3
#  [ 40  45]  # = t4
#  [ 50  55]  # 
#  [ 60  65]
#  [ 70  75]
#  [ 80  85]
#  [ 90  95]
#  [100 105]]
# 
# # Samples: 8
# [[[10 15]
#   [20 25]]] => [[30 35]]  # that is, x1 = [t1, t2], y1 = t3
# [[[20 25]
#   [30 35]]] => [[40 45]]  # that is, x2 = [t2, t3], y2 = t4
# [[[30 35]
#   [40 45]]] => [[50 55]]  # and so on....
# [[[40 45]
#   [50 55]]] => [[60 65]]
# [[[50 55]
#   [60 65]]] => [[70 75]]
# [[[60 65]
#   [70 75]]] => [[80 85]]
# [[[70 75]
#   [80 85]]] => [[90 95]]
# [[[80 85]
#   [90 95]]] => [[100 105]]	