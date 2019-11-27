# multivariate one step problem
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# define generator
n_input = 1
generator = TimeseriesGenerator(dataset, out_seq, length=n_input, batch_size=1)
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))

# Results:
# length = 1, batch_size = 1:
# [[[10 15]]] => [[45]]
# [[[20 25]]] => [[65]]
# [[[30 35]]] => [[85]]
# [[[40 45]]] => [[105]]
# [[[50 55]]] => [[125]]
# [[[60 65]]] => [[145]]
# [[[70 75]]] => [[165]]
# [[[80 85]]] => [[185]]
# [[[90 95]]] => [[205]]
#
# length = 1, batch_size = 2:
# [[[10 15]]
# 
#  [[20 25]]] => [[45]
#  [65]]
# [[[30 35]]
# 
#  [[40 45]]] => [[ 85]
#  [105]]
# [[[50 55]]
# 
#  [[60 65]]] => [[125]
#  [145]]
# [[[70 75]]
# 
#  [[80 85]]] => [[165]
#  [185]]
# [[[90 95]]] => [[205]]