import numpy as np
import glob
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
# importing Statistics module
import statistics
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot
import seaborn as sns
import subprocess
    
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font',**{'family':'serif','serif':['Computer Moder Roman']})
# rc('font',**{'family':'sans-serif'})
rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')

def rho_model(x, rp, rm, xmx, xmn, L):
    f1 = (x-xmn)/L;
    f2 = (xmx-x)/L;
    temp = rm + 0.5*(rp-rm)*((np.tanh(f1)+np.tanh(f2)))
    return temp

pp = PdfPages('density_phase_all_new.pdf')
#fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(wspace=0.0)

colors = sns.color_palette("bright", 18)

system = ["C","A","B"]

for sys in range(len(system)):
        
    z = 80
    points = 10
    frame = 0
    xmn = 20
    xmx = 60
    L = 40
    
    #molar fraction
    ave_low_dens1 = []
    ave_high_dens1 = []
    ave_low_dens2 = []
    ave_high_dens2 = []
    ave_low_mf = []
    ave_high_mf = []
    charge = []
    
    if system[sys] == "A":
        # x1 = [0.0,0.045084736,0.179133699,0.322097085,0.455418087,0.590205818,0.740623291,0.834801253,0.930473267,1.0]
        # x1 = [0.0,0.031330097,0.129375109,0.247903324,0.362342881,0.48026658,0.63452882,0.744281951,0.879920375,1.0]
        x1 = [0.0,0.048737372,0.189931184,0.335212726,0.469114793,0.603581663,0.75157274,0.843028938,0.934411407,1.0]

    if system[sys] == "B":
        # x1 = [0.0, 0.0826875, 0.173, 0.25375, 0.365625, 0.516375, 0.7274375, 0.794625, 0.911875, 1.0]
        # x1 = [0.0,0.045021821,0.09056877,0.129794898,0.323266648,0.611306602,0.709977501,0.803155751,0.893806039,1.0]
        x1 = [0.0,0.097279162,0.237636375,0.383151956,0.468765336,0.582165072,0.696122302,0.791004434,0.88404733,1.0]
    if system[sys] == "C":
        # x1 = [0.0,0.052470792,0.104546096,0.14493233,0.323266648,0.568879292,0.660687569,0.757694537,0.859473802,1.0]
        # x1 = [0.0,0.074171709,0.139018029,0.178522327,0.323266648,0.454601195,0.489142897,0.548010537,0.631512088,0.744281951,0.879920375,1.0]
        x1 = [0.0,0.045021821,0.09056877,0.129794898,0.323266648,0.611306602,0.709977501,0.803155751,0.893806039,1.0]
    for n in range(points):
        # mf = 0.1 + (n * 0.10)
        mf = x1[n]
        charge.append(mf)
        mf_label = "{:.3f}".format(mf)
        mf = str(mf)
        xvals = []
        dens  = []  #np.zeros((100,4))
        stdev = []  #np.zeros((100,4))
        
        #read in the file line by line
        if system[sys] == "A":
            density = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new3/x1="+str(mf)+"/density_1.T-0.77.data"
        if system[sys] == "B":
            density = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new2/x1="+str(mf)+"/density_1.T-0.77.data"
        if system[sys] == "C":
            density = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new3/x1="+str(mf)+"/density_1.T-0.77.data"
        
        all_density = []  # empty lists for each column
        
        with open(density, "r") as file:
        
            lines = file.readlines()
            
            for i, line in enumerate(lines):
        
                line = line.strip()
               
                if line.startswith("#"):  # skip comment lines
                    continue
        
                values = line.split()
                values = [float(value) for value in values]
                if values[0] > 1200:
                    # print(values[0])
                    skip = True
                    continue
                all_density.append(values)
           
        
        block_size = 1200  # Number of lines in each block
        num_blocks = len(all_density) // block_size  # Total number of blocks
        
        blocks = []
        for i in range(num_blocks):
            # print(i)
            if i > frame:
                start = i * block_size
                end = start + block_size
                lines = all_density[start:end]
        
                blocks.append(lines)
        
        ave_block = np.zeros((1200, 4))
        
        low_dens = []
        high_dens = []
        j = 0
        for block in blocks:
            dens = [row[3] for row in block]
            xvals = [row[1]*80 for row in block]
            stdev = np.std(dens, axis=0)
            stdev = np.full_like(dens,stdev)
            #fit the curve        
            #don't shoot the messenger
            #https://gsalvatovallverdu.gitlab.io/python/curve_fit/
            rm = max(dens)
            rp = min(dens)
            popt, pcov = curve_fit(
                f=rho_model,
                xdata=xvals,
                ydata=dens,
                p0=(rp,rm,xmn,xmx,L),
                sigma=stdev,
                maxfev = 1000000
                )
            low_dens.append(abs(popt[1]))
            high_dens.append(abs(popt[0]))
            # Define the range of x-values for the fitted curve
            x_fit = np.linspace(min(xvals), max(xvals), 100)  # Adjust the number of points as needed
            # Calculate the y-values for the fitted curve using the optimized parameters
            y_fit = rho_model(x_fit, *popt)
            ave_block += np.array(block)
            j += 1
        
        ave_block /= len(blocks)
        ave_dens = [row[3] for row in ave_block]
        xvals = [row[1]*80 for row in ave_block]    
        # Convert the array to a NumPy array
        numpy_array = np.array(blocks)
        # Extract the second element from each row
        second_elements = numpy_array[:, :, 3]
        # Calculate the standard deviation of the second elements along the axis 0
        standard_deviations = np.std(second_elements, axis=0)
        xvals = xvals - np.min(xvals)
        x_fit = x_fit - np.min(x_fit)
        print(mf)
        # if mf == str(x1[1]):
            # axes[0,0].plot(xvals,ave_dens,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label))
            # axes[0,0].plot(x_fit, y_fit, color=colors[0], label='x$_{1}$ = '+str(mf_label))
            #axes[0,0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
    #    if mf == str(x1[1]):
            #axes[0,0].plot(xvals,ave_dens,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label))
            # axes[0,0].plot(x_fit, y_fit, color=colors[1], label='x$_{1}$ = '+str(mf_label))
            #axes[0,0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[5]):
            #axes[0,0].plot(xvals,ave_dens,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label))
            # axes[0,0].plot(x_fit, y_fit, color=colors[2], label='x$_{1}$ = '+str(mf_label))
            #axes[0,0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[7]):
            #axes[0,0].plot(xvals,ave_dens,'o',color=colors[3], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)) 
            # axes[0,0].plot(x_fit, y_fit, color=colors[3], label='x$_{1}$ = '+str(mf_label))
            #axes[0,0].fill_between(xvals, dens - standard_deviations, ave_dens + standard_deviations, color=colors[1], alpha=0.5)
        # if mf == str(x1[8]):
            # axes[0,0].plot(xvals,ave_dens,'o',color=colors[8], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label))
            # axes[0,0].plot(x_fit, y_fit, color=colors[4], label='x$_{1}$ = '+str(mf_label))
            #axes[0,0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        # if mf == '1.0':
        #     axes[0].plot(xvals,ave_dens,'o',color=colors[5], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[5], label='q = '+str(q))
        #     axes[0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
    
        # axes[0,0].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='16', loc='upper center', handlelength=0.4)
        # axes[0,0].set_xlabel(r'$\mathbf{\textit{z}\ (\sigma)}$',fontsize=40)
        # axes[0,0].set_ylabel(r'$\mathbf{\rho}$',fontsize=40)
        # axes[0,0].set_xlabel(r'$\mathbf{\textit{z}\ (\sigma)}$',fontsize=40)
        # axes[0,0].set_xticks(np.arange(0, max(xvals)*1.1, 40))
    
        high_dens = [x for x in high_dens if x <= 2*max(y_fit)]
        low_dens = [low_dens[i] for i, x in enumerate(high_dens) if x <= 2 * max(y_fit)]
    
        stdev_low_dens = np.std(ave_dens[40:60])
        stdev_high_dens = np.std(ave_dens[0:30])
        mean_low_dens1 = np.mean([np.mean(ave_dens[500:700]), np.mean(ave_dens[500:700])])
        mean_high_dens1 = np.mean([np.mean(ave_dens[0:200]), np.mean(ave_dens[1000:1200])])
        ave_low_dens1.append(mean_low_dens1)
        ave_high_dens1.append(mean_high_dens1)
    
        xvals = []
        dens  = []  #np.zeros((100,4))
        stdev = []  #np.zeros((100,4))
        
        #read in the file line by line
        if mf not in ["0.0", "1.0"]:
            print('den2 =',mf)
            if system[sys] == "A":
                density = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new3/x1="+str(mf)+"/density_2.T-0.77.data"
            if system[sys] == "B":
                density = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new2/x1="+str(mf)+"/density_2.T-0.77.data"
            if system[sys] == "C":
                density = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new3/x1="+str(mf)+"/density_2.T-0.77.data"
        all_density = []  # empty lists for each column
        
        with open(density, "r") as file:
        
            lines = file.readlines()
            
            for i, line in enumerate(lines):
        
                line = line.strip()
               
                if line.startswith("#"):  # skip comment lines
                    continue
        
                values = line.split()
                values = [float(value) for value in values]
                if values[0] > 1200:
                    # print(values[0])
                    skip = True
                    continue
                all_density.append(values)
           
        
        block_size = 1200  # Number of lines in each block
        num_blocks = len(all_density) // block_size  # Total number of blocks
        
        blocks = []
        for i in range(num_blocks):
            # print(i)
            if i > frame:
                start = i * block_size
                end = start + block_size
                lines = all_density[start:end]
        
                blocks.append(lines)
        
        ave_block = np.zeros((1200, 4))
        
        low_dens = []
        high_dens = []
        j = 0
        for block in blocks:
            dens = [row[3] for row in block]
            xvals = [row[1]*80 for row in block]
            stdev = np.std(dens, axis=0)
            stdev = np.full_like(dens,stdev)
            #fit the curve        
            #don't shoot the messenger
            #https://gsalvatovallverdu.gitlab.io/python/curve_fit/
            rm = max(dens)
            rp = min(dens)
            popt, pcov = curve_fit(
                f=rho_model,
                xdata=xvals,
                ydata=dens,
                p0=(rp,rm,xmn,xmx,L),
                sigma=stdev,
                maxfev = 100000
                )
            low_dens.append(abs(popt[1]))
            high_dens.append(abs(popt[0]))
            # Define the range of x-values for the fitted curve
            x_fit = np.linspace(min(xvals), max(xvals), 100)  # Adjust the number of points as needed
            # Calculate the y-values for the fitted curve using the optimized parameters
            y_fit = rho_model(x_fit, *popt)
            ave_block += np.array(block)
            j += 1
        
        ave_block /= len(blocks)
        ave_dens = [row[3] for row in ave_block]
        xvals = [row[1]*80 for row in ave_block]    
        # Convert the array to a NumPy array
        numpy_array = np.array(blocks)
        # Extract the second element from each row
        second_elements = numpy_array[:, :, 3]
        # Calculate the standard deviation of the second elements along the axis 0
        standard_deviations = np.std(second_elements, axis=0)
        xvals = xvals - np.min(xvals)
        x_fit = x_fit - np.min(x_fit)
        rounded_mf_label = "{:.3f}".format(1-float(mf_label))
        # if mf == str(x1[1]):
            # axes[0,1].plot(xvals,ave_dens,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{2}$ = '+str(rounded_mf_label))
            # axes[0,1].plot(x_fit, y_fit, color=colors[0], label='x$_{2}$ = '+str(rounded_mf_label))
            #axes[0,1].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[3]):
            #axes[0,1].plot(xvals,ave_dens,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{2}$ = '+str(rounded_mf_label))
            # axes[0,1].plot(x_fit, y_fit, color=colors[1], label='x$_{2}$ = '+str(rounded_mf_label))
            #axes[0,1].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[5]):
            #axes[0,1].plot(xvals,ave_dens,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{2}$ = '+str(rounded_mf_label))
            # axes[0,1].plot(x_fit, y_fit, color=colors[2], label='x$_{2}$ = '+str(rounded_mf_label))
            #axes[0,1].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[7]):
            #axes[0,1].plot(xvals,ave_dens,'o',color=colors[3], markersize=8, fillstyle='none', label='x$_{2}$ = '+str(rounded_mf_label)) 
            # axes[0,1].plot(x_fit, y_fit, color=colors[3], label='x$_{2}$ = '+str(rounded_mf_label))
            #axes[0,1].fill_between(xvals, dens - standard_deviations, ave_dens + standard_deviations, color=colors[1], alpha=0.5)
        # if mf == str(x1[8]):
            # axes[0,1].plot(xvals,ave_dens,'o',color=colors[8], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label))
            # axes[0,1].plot(x_fit, y_fit, color=colors[4], label='x$_{2}$ = '+str(rounded_mf_label))
            #axes[0,1].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        # if mf == '1.0':
        #     axes[0].plot(xvals,ave_dens,'o',color=colors[5], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[5], label='q = '+str(q))
        #     axes[0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
    
        # axes[0,1].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='16', loc='upper center', handlelength=0.4)
        # axes[0,1].set_xlabel(r'$\mathbf{\textit{z}\ (\sigma)}$',fontsize=40)
        # axes[0,1].set_ylabel(r'$\mathbf{\rho}$',fontsize=40)
        # axes[0,1].set_xticks(np.arange(0, max(xvals)*1.1, 40))
    
        high_dens = [x for x in high_dens if x <= 2*max(y_fit)]
        low_dens = [low_dens[i] for i, x in enumerate(high_dens) if x <= 2 * max(y_fit)]
    
        stdev_low_dens = np.std(ave_dens[500:700])
        stdev_high_dens = np.std(ave_dens[0:200])
        mean_low_dens2 = np.mean([np.mean(ave_dens[500:700]), np.mean(ave_dens[500:700])])
        mean_high_dens2 = np.mean([np.mean(ave_dens[0:200]), np.mean(ave_dens[1000:1200])])
        ave_low_dens2.append(mean_low_dens2)
        ave_high_dens2.append(mean_high_dens2)
        if mf in ["0.0"]:
            mean_low_mf = 1.0
            mean_high_mf = 1.0
            ave_low_mf.append(mean_low_mf)
            ave_high_mf.append(mean_high_mf)
        if mf not in ["0.0", "1.0"]:
            mean_low_mf = mean_low_dens2/(mean_low_dens1+mean_low_dens2)
            mean_high_mf = mean_high_dens2/(mean_high_dens1+mean_high_dens2)
            ave_low_mf.append(mean_low_mf)
            ave_high_mf.append(mean_high_mf)
        if mf in ["1.0"]:
            mean_low_mf = 0.0
            mean_high_mf = 0.0
            ave_low_mf.append(mean_low_mf)
            ave_high_mf.append(mean_high_mf)
    
    # #plot the density model with the fit parameters
    
    # #commented out for debugging, will plot fitted data
    # axes[1].plot(ave_low_dens,charge,'-o')
    # axes[1].plot(ave_high_dens,charge,'-s')
    # axes[1].fill_between(ave_low_dens-stdev_low_dens, ave_low_dens+stdev_low_dens, charge, alpha=1)
    # # axes[1].fill_between(ave_high_dens-stdev_high_dens, ave_high_dens+stdev_high_dens, charge, alpha=1)
    # axes.errorbar(ave_low_dens,charge, xerr=stdev_low_dens,fmt='-o', color=colors[2], ecolor=colors[2], markersize=12, fillstyle='none', label = 'Supernatant')
    # axes.errorbar(ave_high_dens,charge, xerr=stdev_high_dens,fmt='-s', color=colors[3], ecolor=colors[3], markersize=12, fillstyle='none', label = 'Coacervate')
    # axes.legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper center', handlelength=0.4)
    # axes.set_xlabel(r'$\mathbf{\rho}$',fontsize=40)
    # axes.set_ylabel(r'$\mathbf{\textit{q}}$',fontsize=40)
    # pyplot.show()
    # pyplot.savefig('PhaseDiagram.png')
    
    #presure
    z = 80
    ave_low_press = []
    ave_high_press = []
    ave_press_all = []
    charge = []
    for n in range(points):
        # mf = 0.1 + (n * 0.10)
        mf = x1[n]
        charge.append(mf)
        # mf = "{:.7f}".format(mf)
        mf = str(mf)
    
        xvals = []
        press  = []  #np.zeros((100,4))
        stdev = []  #np.zeros((100,4))
        
        #read in the file line by line
        if system[sys] == "A":
            pressure = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new3/x1="+str(mf)+"/pzz.txt"
        if system[sys] == "B":
            pressure = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new2/x1="+str(mf)+"/pzz.txt"
        if system[sys] == "C":
            pressure = "/scratch365/mfarshad/gpr/md/system"+str(system[sys])+"/lj_new3/x1="+str(mf)+"/pzz.txt"
        all_pressure = []  # empty lists for each column
        
        with open(pressure, "r") as file:
        
            lines = file.readlines()
            
            for i, line in enumerate(lines):
        
                line = line.strip()
               
                if line.startswith("#"):  # skip comment lines
                    continue
        
                values = line.split()
                values = [float(value) for value in values]
                if values[0] > 1200:
                    # print(values[0])
                    skip = True
                    continue
                all_pressure.append(values)
           
        
        block_size = 1200  # Number of lines in each block
        num_blocks = len(all_pressure) // block_size  # Total number of blocks
        
        blocks = []
        for i in range(num_blocks):
            # print(i)
            if i > frame:
                start = i * block_size
                end = start + block_size
                lines = all_pressure[start:end]
        
                blocks.append(lines)
        
        ave_block = np.zeros((1200, 2))
        
        low_press = []
        high_press = []
        j = 0
        for block in blocks:
            press = [row[1] for row in block]
            xvals = [row[0]*80/100 for row in block]
            stdev = np.std(dens, axis=0)
            stdev = np.full_like(dens,stdev)
            #fit the curve        
            #don't shoot the messenger
            #https://gsalvatovallverdu.gitlab.io/python/curve_fit/
            # popt, pcov = curve_fit(
            #     f=rho_model,
            #     xdata=xvals,
            #     ydata=press,
            #     p0=(0,0.25,20,40,10),
            #     sigma=stdev,
                # maxfev = 100000
                # )
            low_press.append(abs(popt[1]))
            high_press.append(abs(popt[0]))
            # Define the range of x-values for the fitted curve
            x_fit = np.linspace(min(xvals), max(xvals), 100)  # Adjust the number of points as needed
            # Calculate the y-values for the fitted curve using the optimized parameters
            y_fit = rho_model(x_fit, *popt)
            ave_block += np.array(block)
            j += 1
        
        ave_block /= len(blocks)
        ave_press = [row[1] for row in ave_block]
        xvals = [row[0]*80/1200 for row in ave_block]    
        # Convert the array to a NumPy array
        numpy_array = np.array(blocks)
        # Extract the second element from each row
        second_elements = numpy_array[:, :, 1]
        # Calculate the standard deviation of the second elements along the axis 0
        standard_deviations = np.std(second_elements, axis=0)
        xvals = xvals - np.min(xvals)
        x_fit = x_fit - np.min(x_fit)
        # if mf == str(x1[1]):
            # axes[1,0].plot(xvals,ave_press,'o-',color=colors[0], markersize=8, fillstyle='none')
            # axes[1].plot(x_fit, y_fit, color=colors[0], label='x1 = '+str(mf))
            # axes[1].fill_between(xvals, ave_press - standard_deviations, ave_press + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[3]):
        #    axes[1,0].plot(xvals,ave_press,'o-',color=colors[1], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[1], label='q = '+str(q))
        #     axes[0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[4]):
            #axes[1,0].plot(xvals,ave_press,'o-',color=colors[2], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[2], label='q = '+str(q))
        #     axes[0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        #if mf == str(x1[5]):
        #    axes[1,0].plot(xvals,ave_press,'o-',color=colors[3], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[3], label='q = '+str(q))
        #     axes[0].fill_between(xvals, dens - standard_deviations, ave_dens + standard_deviations, color=colors[1], alpha=0.5)
        #if mf == str(x1[6]):
        #    axes[1,0].plot(xvals,ave_press,'o-',color=colors[4], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[4], label='q = '+str(q))
        #     axes[0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
        # if mf == str(x1[8]):
            # axes[1,0].plot(xvals,ave_press,'o-',color=colors[8], markersize=8, fillstyle='none')
        #     axes[0].plot(x_fit, y_fit, color=colors[5], label='q = '+str(q))
        #     axes[0].fill_between(xvals, ave_dens - standard_deviations, ave_dens + standard_deviations, color=colors[0], alpha=0.5)
    
        # axes[1,0].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
        # axes[1,0].set_ xlabel(r'$\mathbf{\textit{z}\ (\sigma)}$',fontsize=40)
        # axes[1,0].set_ylabel(r'$\mathbf{\textit{p}}$',fontsize=40)
        # axes[1,0].set_xlabel(r'$\mathbf{\textit{z}\ (\sigma)}$',fontsize=40)
        # axes[1,0].set_xticks(np.arange(0, max(xvals)*1.1, 40))
        high_dens = [x for x in high_press if x <= 2*max(y_fit)]
        low_dens = [low_press[i] for i, x in enumerate(high_dens) if x <= 2 * max(y_fit)]
    
        stdev_low_press = np.std(low_press)
        stdev_high_press = np.std(high_press)
        mean_low_press = np.mean(low_press)
        mean_high_press = np.mean(high_press)
        ave_low_press.append(mean_low_press)
        ave_high_press.append(mean_high_press)
        # Assuming ave_press is your array of pressure values    
        # Step 1: Find the minimum pressure
        min_press = np.min(ave_press)
        # Step 2: Determine indices of areas with pressure smaller than 1.2 times min_press
        # smaller_indices = np.where(ave_press < (1.2 * ave_press[0]))[0]
        # smaller_indices = [i for i, press in enumerate(ave_press) if (press < ave_press[600]+(500.25*(max(ave_press)-min(ave_press))) and press > ave_press[600]-(500.25*(max(ave_press)-min(ave_press))))]
        smaller_indices = [i for i, press in enumerate(ave_press) if (press < np.mean(ave_press[500:700])*(1.05) and press > np.mean(ave_press[500:700])*(0.95))]
    
        # Step 3: Extract pressure values from smaller areas
        # smaller_pressures = ave_press[smaller_indices]
        smaller_pressures = [ave_press[i] for i in smaller_indices]
    
        # Step 4: Compute the mean of the extracted pressure values
        mean_press_all = np.mean(smaller_pressures)
        # mean_press_all = np.mean([np.mean(ave_press[0:200]), np.mean(ave_press[500:700]), np.mean(ave_press[1000:1200])])
        ave_press_all.append(mean_press_all)  
        reduced_pressure = [press / 0.098 for press in ave_press_all]
    
    reduced_pressure[0], reduced_pressure[-1] = reduced_pressure[-1], reduced_pressure[0]
    #phase diagram
    axes[sys].plot(ave_low_mf,reduced_pressure,'o-',color=colors[16], markersize=8, fillstyle='none')
    axes[sys].plot(ave_high_mf,reduced_pressure,'o-',color=colors[16], markersize=8, fillstyle='none')
    if sys==0:
        axes[sys].set_ylabel(r'$\mathbf{\textit{p}/{\textit{p}_{\mathbf{c,1}}}}$', fontsize=40)
    axes[sys].set_xlabel(r'$\mathbf{\textit{x}}_{\mathbf{2}}$', fontsize=40)
    
    #experimental phase diagram: 
    
    if system[sys] == "A":
        p_ex = [0.20868179293253003,0.202370717,0.188678442,0.172100447,0.160586508,0.146636306,0.133636944,0.121968078,0.1136412,0.10748031916598126]
        x_l = [1,0.950305434,0.807122631,0.661066554,0.526865698,0.392332397,0.245059335,0.154390943,0.06430433,0]
        x_v = [1,0.973279746,0.884368563,0.76893303,0.655373436,0.537195234,0.379788541,0.266525773,0.12530202,0]

        axes[sys].plot(x_l,p_ex,'o-',color=colors[2], markersize=8, fillstyle='none')
        axes[sys].plot(x_v,p_ex,'o-',color=colors[2], markersize=8, fillstyle='none')
        axes[sys].set_xticks(np.arange(0, 1.1, 0.2))
        axes[sys].set_yticks([])
        # axes[sys].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper left', handlelength=0.4)
    if system[sys] == "B":
        p_ex = [0.20868179293253003,0.180884702,0.137081795,0.101918755,0.087189516,0.075373708,0.076127422,0.082919796,0.093934161,0.10748031916598126]
        x_l = [1,0.899494202,0.757676988,0.613012238,0.528660938,0.417834211,0.304520046,0.210174675,0.117120112,0]
        x_v = [1,0.988836998,0.942883101,0.827485915,0.698526927,0.417889611,0.255604435839542,0.12752968,0.046286294,0]

        axes[sys].plot(x_l,p_ex,'o-',color=colors[2], markersize=8, fillstyle='none')
        axes[sys].plot(x_v,p_ex,'o-',color=colors[2], markersize=8, fillstyle='none')
        axes[sys].set_xticks(np.arange(0, 1.1, 0.2))
        axes[sys].set_yticks([])
        # axes[sys].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper left', handlelength=0.4)
    if system[sys] == "C":
        p_ex = [0.20868179293253003,0.221006365,0.232458817,0.24009268,0.247758586,0.236148131,0.225738719,0.208190446,0.17733346,0.10748031916598126]
        x_l = [1,0.956896852,0.912972028,0.873984221,0.676733352,0.377281368,0.275964598,0.183080448,0.095356141,0]
        x_v = [1,0.916460668,0.84346385,0.802561121,0.676733352,0.599238209,0.574204959,0.511214478,0.413657918,0]
        axes[sys].plot(x_l,p_ex,'o-',color=colors[2], markersize=8, fillstyle='none')
        axes[sys].plot(x_v,p_ex,'o-',color=colors[2], markersize=8, fillstyle='none')
        axes[sys].set_xticks(np.arange(0, 1.1, 0.2))
        
        axes[sys].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='lower right', handlelength=0.4)
    
    axes[sys].set_ylim(0.06,0.27)

    # #plot the density model with the fit parameters
    
    # #commented out for debugging, will plot fitted data
    # axes[1].plot(ave_low_dens,charge,'-o')
    # axes[1].plot(ave_high_dens,charge,'-s')
    # axes[1].fill_between(ave_low_dens-stdev_low_dens, ave_low_dens+stdev_low_dens, charge, alpha=1)
    # # axes[1].fill_between(ave_high_dens-stdev_high_dens, ave_high_dens+stdev_high_dens, charge, alpha=1)
    # axes.errorbar(ave_low_dens,charge, xerr=stdev_low_dens,fmt='-o', color=colors[5], ecolor=colors[2], markersize=12, fillstyle='none', label = 'Supernatant')
    # axes.errorbar(ave_high_dens,charge, xerr=stdev_high_dens,fmt='-s', color=colors[9], ecolor=colors[3], markersize=12, fillstyle='none', label = 'Coacervate')
    # # axes.legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper center', handlelength=0.4)
    # axes.set_xlabel(r'$\mathbf{\rho}$',fontsize=40)
    # axes.set_ylabel(r'$\mathbf{\textit{q}}$',fontsize=40)
    # pyplot.show()
    # pyplot.savefig('PhaseDiagram.png')
    # axes[0].set_ylim(-0.1,0.4)
    # axes[1].set_ylim(-0.1,0.4)
    if system[sys] == "A":
        axes[sys].set_title(r'$\mathbf{\xi = 1}$', fontsize=24, fontweight='bold')  
    if system[sys] == "B":
        axes[sys].set_title(r'$\mathbf{\xi = 1.2}$', fontsize=24, fontweight='bold')  
    if system[sys] == "C":
        axes[sys].set_title(r'$\mathbf{\xi = 0.85}$', fontsize=24, fontweight='bold')  

plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()
