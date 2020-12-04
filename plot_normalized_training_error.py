
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%%

implementation_1 = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\out1.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\out2.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\out3.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\out4.txt"
					]

#%%

implementation_2 = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\out_impala1.txt",
				   r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\out_impala2.txt",
				   r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\out_impala3.txt",
				   r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\out_impala4.txt",
				   r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\out_impala5.txt",
				   r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\out_impala6.txt"
				   ]

#%%

implementation_3 = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack1.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack2.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack3.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack4.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack5.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack6.txt"
					]

#%%

implementation_4 = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack_background1.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack_background2.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack_background3.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack_background4.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack_background5.txt",
					r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\out_impala_stack_background6.txt"
					]

#%%

all_implementations = [implementation_1,
					   implementation_2,
					   implementation_3,
					   implementation_4]

#%%

def read_file(path):
	x = np.zeros(977)
	y = np.zeros(977)
	i = 0
	with open(path, 'r') as infile:
		for line in infile:
			if "Step" in line:
				tmp = line.split("\tMean reward: ")
				x[i] = float(tmp[0].replace("Step: ",""))
				y[i] = float(tmp[1])
				i += 1
	return x,y


#%%

feature_settings = np.array([256,256*2,256*3,256*4,256*5,256*6])

#%% Plotting

font = {'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

for i in range(4):
	plt.figure()
	plt.title("Implementation {}".format(i+1))
	for j in range(len(all_implementations[i])):
		x,y = read_file(all_implementations[i][j])
		plt.plot(x,y,label="Feature_dim: {}".format(feature_settings[j]))
	plt.ylabel("Normalized training error")
	plt.xlabel("Timestep")
	plt.ylim([4,28])
	plt.legend()
	plt.grid()
	plt.show()




