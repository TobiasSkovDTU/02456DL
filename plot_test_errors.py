

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#%%

implementation_1_without_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint1_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint2_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint3_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint4_evaluation.txt"
									]

implementation_1_with_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint1_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint2_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint3_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\baseline_tests\checkpoint4_evaluation_generalization.txt"
								 ]


#%%

implementation_2_without_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala1_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala2_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala3_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala4_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala5_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala6_evaluation.txt"
									]

implementation_2_with_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala1_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala2_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala3_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala4_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala5_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week11_Results\impala_tests\checkpoint_impala6_evaluation_generalization.txt"
								 ]

#%%

implementation_3_without_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack1_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack2_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack3_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack4_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack5_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack6_evaluation.txt"]

implementation_3_with_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack1_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack2_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack3_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack4_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack5_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack6_evaluation_generalization.txt"]

#%%

implementation_4_without_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background1_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background2_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background3_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background4_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background5_evaluation.txt",
									r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background6_evaluation.txt"]

implementation_4_with_general = [r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background1_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background2_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background3_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background4_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background5_evaluation_generalization.txt",
								 r"C:\Users\Tubsp\Documents\GitHub\02456DL\Week12_Results\checkpoint_impala_stack_background6_evaluation_generalization.txt"]



#%%


def read_file(path):
	fixed_length_inf_lives = 0
	inf_length_one_life = 0
	max_steps_survived = 0
	with open(path, 'r') as infile:
		for line in infile:
			if "Average return (fixed simulation lenght): " in line:
				fixed_length_inf_lives = float(line.replace("Average return (fixed simulation lenght): ",""))
				
			elif "Average return (running until death): " in line:
				inf_length_one_life = float(line.replace("Average return (running until death): ",""))
				
			elif "Longest run in number of steps " in line:
				max_steps_survived = float(line.replace("Longest run in number of steps ",""))
				
	return np.array([fixed_length_inf_lives, inf_length_one_life, max_steps_survived])

#%%

implementations_without_general = [implementation_1_without_general,
								   implementation_2_without_general,
								   implementation_3_without_general,
								   implementation_4_without_general]

implementations_with_general = [implementation_1_with_general,
								implementation_2_with_general,
								implementation_3_with_general,
								implementation_4_with_general]

feature_settings = np.array([256,256*2,256*3,256*4,256*5,256*6])


#%%
font = {'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

for i in range(4):
	
	cur_len = len(implementations_without_general[i])
	
	y_without_gen = np.zeros((3,cur_len))

	y_with_gen = np.zeros((3,cur_len))

	for j in range(cur_len):
		
		y_without_gen[:,j] = read_file(implementations_without_general[i][j])
		y_with_gen[:,j] = read_file(implementations_with_general[i][j])
		
	#Testing errors
	
	plt.figure()
	plt.plot(feature_settings[0:cur_len],y_without_gen[0,:],ls = "--",
		  label = "wo gen, test error 1",marker='o',c="green")
	plt.plot(feature_settings[0:cur_len],y_without_gen[1,:],
		  label = "wo gen, test error 2",marker='s',c="green")
	plt.plot(feature_settings[0:cur_len],y_with_gen[0,:],ls = "--",
		  label = "w gen, test error 1",marker='o', c="red")
	plt.plot(feature_settings[0:cur_len],y_with_gen[1,:],
		  label = "w gen, test error 2",marker='s', c="red")
	plt.title("Implementation {}".format(i+1))
	plt.ylabel("Testing errors")
	plt.xlabel("Different feature dimension settings")
	plt.xticks(feature_settings[0:cur_len],feature_settings[0:cur_len])
	plt.ylim([8,22])
	plt.legend()
	plt.grid()
	plt.show()
	
	
	#Survivability
	plt.figure()
	plt.plot(feature_settings[0:cur_len],y_without_gen[2,:],
		  label = "wo gen",marker='>',c="green")
	plt.plot(feature_settings[0:cur_len],y_with_gen[2,:],
		  label = "w gen",marker='>', c="red")
	plt.title("Implementation {}".format(i+1))
	plt.ylabel("Maximum survived steps")
	plt.xlabel("Different feature dimension settings")
	plt.xticks(feature_settings[0:cur_len],feature_settings[0:cur_len])
	plt.ylim([300,600])
	plt.legend()
	plt.grid()
	plt.show()
	
