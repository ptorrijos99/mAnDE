# Create a script that read every mAnDE.o* file and add the data of the mean of the mSPnDEs size, the mean of the vars in each mSPnDE, the mean of max vars in each mSPnDE, and the mean of min vars in each mSPnDE, to the end of the line of a csv file
# This is an example of a mAnDE.o* file:
'''
experiment_results_Breast_GSE38959.arff_mAnDE_2_3_false_200_none_RandomTree_1_RF_100_0.15.csv

mSPnDEs,593,1.4468802698145025,4.0,0.0
mSPnDEs,616,1.4545454545454546,3.0,0.0
mSPnDEs,598,1.4046822742474916,3.0,0.0
mSPnDEs,607,1.4365733113673806,3.0,0.0
mSPnDEs,625,1.4496,3.0,0.0
mSPnDEs,602,1.4086378737541527,4.0,0.0
mSPnDEs,571,1.3730297723292468,3.0,0.0
mSPnDEs,590,1.4305084745762713,4.0,0.0
mSPnDEs,569,1.4130052724077329,3.0,0.0
mSPnDEs,614,1.4169381107491856,3.0,0.0
mSPnDEs,597,1.4137353433835846,3.0,0.0
mSPnDEs,641,1.4851794071762872,3.0,0.0
mSPnDEs,547,1.3564899451553931,3.0,0.0
mSPnDEs,583,1.4236706689536878,3.0,0.0
mSPnDEs,561,1.3511586452762923,3.0,0.0
mSPnDEs,618,1.4336569579288025,3.0,0.0
mSPnDEs,563,1.3676731793960923,3.0,0.0
mSPnDEs,562,1.3736654804270463,4.0,0.0
mSPnDEs,535,1.3495327102803738,3.0,0.0
mSPnDEs,584,1.393835616438356,3.0,0.0
mSPnDEs,599,1.4257095158597664,3.0,0.0
mSPnDEs,587,1.4207836456558773,3.0,0.0
mSPnDEs,618,1.4142394822006472,3.0,0.0
mSPnDEs,595,1.4252100840336135,3.0,0.0
mSPnDEs,594,1.4006734006734007,3.0,0.0
mSPnDEs,582,1.3917525773195876,3.0,0.0
mSPnDEs,524,1.316793893129771,3.0,0.0
mSPnDEs,604,1.4403973509933774,4.0,0.0
mSPnDEs,562,1.3345195729537367,3.0,0.0
mSPnDEs,642,1.4361370716510904,3.0,0.0
mSPnDEs,553,1.403254972875226,3.0,0.0
mSPnDEs,553,1.345388788426763,3.0,0.0
mSPnDEs,541,1.3715341959334566,3.0,0.0
mSPnDEs,566,1.3851590106007068,3.0,0.0
mSPnDEs,592,1.412162162162162,3.0,0.0
mSPnDEs,598,1.4180602006688963,3.0,0.0
mSPnDEs,560,1.3785714285714286,3.0,0.0
mSPnDEs,541,1.3197781885397413,3.0,0.0
mSPnDEs,348,1.0172413793103448,3.0,0.0
mSPnDEs,608,1.4539473684210527,3.0,0.0
mSPnDEs,574,1.3658536585365855,3.0,0.0
mSPnDEs,576,1.40625,3.0,0.0
mSPnDEs,558,1.3870967741935485,4.0,0.0

'''
# Mean of mSPnDEs size: (593+616+625+602+571+590+569+614+597+641+547+583+561+618+563+562+535+584+599+587+618+595+594+582+524+604+562+642+553+553+541+566+592+598+560+541+348+608+574+576+558)/43 = 576.9767441860465
# Mean of mSPnDEs var: (1.4468802698145025+1.4545454545454546+1.4046822742474916+1.4365733113673806+1.4496+1.4086378737541527+1.3730297723292468+1.4305084745762713+1.4130052724077329+1.4169381107491856+1.4137353433835846+1.4851794071762872+1.3564899451553931+1.4236706689536878+1.3511586452762923+1.4336569579288025+1.3676731793960923+1.3736654804270463+1.3495327102803738+1.393835616438356+1.4257095158597664+1.4207836456558773+1.4142394822006472+1.4252100840336135+1.4006734006734007+1.3917525773195876+1.316793893129771+1.4403973509933774+1.3345195729537367+1.4361370716510904+1.403254972875226+1.345388788426763+1.3715341959334566+1.3851590106007068+1.412162162162162+1.4180602006688963+1.3785714285714286+1.3197781885397413+1.0172413793103448+1.4539473684210527+1.3658536585365855+1.40625+1.3870967741935485)/43 = 1.3986190476190476
# Mean of mSPnDEs max var: (4.0+3.0+3.0+3.0+3.0+4.0+3.0+4.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+4.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+4.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0+3.0)/43 = 3.1627906976744187
# Mean of mSPnDEs min var: (0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0)/43 = 0.0

# This is an example of the original experiment_results_Breast_GSE38959.arff_mAnDE_2_3_false_200_none_RandomTree_2_RF_100_0.15.csv file:
'''
bbdd,algorithm,seed,folds,discretized,nTrees,featureSelection,baseClass,n,ensemble,bagSize,porNB,score,fm,precision,recall,probAciertos,probPredFallos,probRealFallos,time(s)
Breast_GSE38959.arff,mAnDE,2,43,false,200,none,RandomTree,1,RF,100,0.15,97.67441860465117,0.9718032786885247,0.9838709677419355,0.9615384615384616,0.8439096971751244,0.9369083406139264,0.06309165938607371,1.1803720930232557
'''
# This is an example of the experiment_results_Breast_GSE38959.arff_mAnDE_2_3_false_200_none_RandomTree_2_RF_100_0.15.csv file afther the changues: 
'''
bbdd,algorithm,seed,folds,discretized,nTrees,featureSelection,baseClass,n,ensemble,bagSize,porNB,score,fm,precision,recall,probAciertos,probPredFallos,probRealFallos,time(s),mAnDEs,var,maxVar,minVar
Breast_GSE38959.arff,mAnDE,2,43,false,200,none,RandomTree,1,RF,100,0.15,97.67441860465117,0.9718032786885247,0.9838709677419355,0.9615384615384616,0.8439096971751244,0.9369083406139264,0.06309165938607371,1.1803720930232557,576.9767441860465,1.3986190476190476,3.1627906976744187,0.0
'''

# So, the code is:
import os
import csv
import numpy as np
from pathlib import Path

# Obtain the list of mAnDE.o* files in the parent directory
files = [str(x) for x in Path('../').glob('mAnDE.o*')]

# For each file
for file in files:
    # Open the file
    with open(file, 'r') as f:
        # Read the file
        reader = f.readlines()

        # Obtain the name of the file that contains the results of the experiment (the first line of the file)
        name = reader[0].split(' ')[-1].strip()

        # Read between the third line and the next empty line all of the results, calculating the four means
        mSPnDEs = []
        mSPnDEsVar = []
        mSPnDEsMaxVar = []
        mSPnDEsMinVar = []

        for line in reader[2:]:
            print(line.split(','))
            if line.split(',')[0] == '\n':
                break
            else:
                mSPnDEs.append(float(line.split(',')[1]))
                mSPnDEsVar.append(float(line.split(',')[2]))
                mSPnDEsMaxVar.append(float(line.split(',')[3]))
                mSPnDEsMinVar.append(float(line.split(',')[4]))

        # Calculate the four means
        mSPnDEsMean = np.mean(mSPnDEs)
        mSPnDEsVarMean = np.mean(mSPnDEsVar)
        mSPnDEsMaxVarMean = np.mean(mSPnDEsMaxVar)
        mSPnDEsMinVarMean = np.mean(mSPnDEsMinVar)

        lines = []

        # Open the file that contains the results of the experiment
        with open('../results/'+name, 'r') as f2:
            # Read the file
            reader = csv.reader(f2, delimiter=',')
            # Create a list with the lines of the file
            lines = list(reader)

            # Add the four means to the last four columns of the file
            lines[0].append('mSPnDEs')
            lines[0].append('vars')
            lines[0].append('maxVars')
            lines[0].append('minVars')

            lines[1].append(mSPnDEsMean)
            lines[1].append(mSPnDEsVarMean)
            lines[1].append(mSPnDEsMaxVarMean)
            lines[1].append(mSPnDEsMinVarMean)

            print(lines[0])
            print(lines[1])
        # Close the file that contains the results of the experiment 
        f2.close()

        # Open the file that contains the results of the experiment
        with open('../results/'+name, 'w', newline='') as f2:
            # Write the file
            writer = csv.writer(f2, delimiter=',')
            writer.writerows(lines)
            
        # Close the file that contains the results of the experiment 
        f2.close()

    # Close the file
    f.close()
    