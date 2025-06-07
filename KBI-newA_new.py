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
#from scipy.integrate import cumtrapz
from scipy import integrate
from MDAnalysis.analysis.rdf import InterRDF
import MDAnalysis as md
from scipy import stats

tinv = lambda p, df: abs(stats.t.ppf(p/2, df))

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

Na = 6.02214e23 
units_factor = Na/1e21                              # From nm3/molecules to cm3/mol

def KBI_Integral(r, rdf):
    """
    KBI Truncating the integral at rmax.

    Args:
        r (array): vector of equispaced radii at which RDF is evaluated [units]
        rdf (array): vector of corresponding RDF values

    Returns:
        KBI (float): KBI value [units**3]
    """
    dr = r[1] - r[0]
    n = len(r)
    KBI = 4 * np.pi * integrate.cumtrapz( ( rdf - 1 ) * r**2 , r, initial = 0)
    return KBI

def KBI_Kruger(r, rdf):
    """
    KBI using Kruger correction for finite size effect.
    Kruger et al. J. Phys. Chem. Lett. 4, 235-238, 2013.

    Args:
        r (array): vector of equispaced radii at which RDF is evaluated [units]
        rdf (array): vector of corresponding RDF values

    Returns:
        KBI (float): KBI value [units**3]
    """
    dr = r[1] - r[0]
    n = len(r)
    KBI = np.zeros(n)
    for i in range(2, n):
        rvec = r[0:i]
        c = 4 * np.pi * rvec**2
        Lmax = max(rvec)      
        x = rvec/(Lmax)
        w = 1 - 3 * x / 2 + x**3 / 2
        h = rdf[0:i] - 1
        KBI[i] = np.trapz(h * c * w , x = rvec, dx = dr)
    return KBI

def KBI_extrapolation(r, KBI, x, y):
    r_inverse = 1/r
    index = (r_inverse >= x) & (r_inverse <= y)
    res = stats.linregress(r_inverse[index], KBI[index])
    G_inf_ij_value = res.intercept
    G_inf_ij_error = tinv(0.95, len(r_inverse[index])) * res.stderr
    return G_inf_ij_value, G_inf_ij_error

correction = 0
if correction==10:
    pp1 = PdfPages('RDF+KBI-corrected_A_new.pdf')
if correction==0:
    pp1 = PdfPages('RDF+KBI_A_new.pdf')
#fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig1, axes = plt.subplots(9, 3, figsize=(15, 45))
# plt.subplots_adjust(wspace=0.0,hspace=0.0)

colors = sns.color_palette("bright", 18)

z = 80
points = 10
frame = 0
xmn = 20
xmx = 60
L = 40

#molar fraction
KBI_1_1_all = []
KBI_1_2_all = []
KBI_2_2_all = []
KBI_1_1_all_corrected = []
KBI_1_2_all_corrected = []
KBI_2_2_all_corrected = []
KBI_1_1_all_corrected_error = []
KBI_1_2_all_corrected_error = []
KBI_2_2_all_corrected_error = []
charge = []
# x1 = [0.0,0.047274072,0.183354641,0.328762083,0.462195728,0.598866151,0.746312697,0.837429587,0.932013532,1.0]
x1 = [0.0, 0.049878508, 0.19483645, 0.341489253, 0.476125023, 0.612307586, 0.757463381, 0.84820193, 0.936649383, 1.0]

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
    rdf_file = "/scratch365/mfarshad/gpr/md/systemA/lj_cube1/x1="+str(mf)+"/rdf.data"
    all_rdf = []  # empty lists for each column
    
    with open(rdf_file, "r") as file:
    
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
            all_rdf.append(values)
       
    
    block_size = 1200  # Number of lines in each block
    num_blocks = len(all_rdf) // block_size  # Total number of blocks
    
    blocks = []
    for i in range(num_blocks):
        # print(i)
        if i > -1:
            start = i * block_size
            end = start + block_size
            lines = all_rdf[start:end]
    
            blocks.append(lines)
    
    if mf != str(x1[0]) and mf != str(x1[-1]):
        ave_block = np.zeros((1200, 8))
    
    if mf == '0.0' or mf == '1.0':
        ave_block = np.zeros((1200, 4))
        
    j = 0
    for block in blocks:
        rdf = [row[3] for row in block]
        xvals = [row[1] for row in block]
        stdev = np.std(rdf, axis=0)
        stdev = np.full_like(rdf,stdev)

        # Define the range of x-values for the fitted curve
        x_fit = np.linspace(min(xvals), max(xvals), 100)  # Adjust the number of points as needed
        # Calculate the y-values for the fitted curve using the optimized parameters
        ave_block += np.array(block)
        j += 1
    
    ave_block /= len(blocks)
    # PDB = "/scratch365/mfarshad/gpr/md/systemA/lj_cube/x1="+str(mf)+"/system.pdb"
    # DUMP = "/scratch365/mfarshad/gpr/md/systemA/lj_cube/x1="+str(mf)+"/md_T0.77.dump"
    # u = md.Universe(PDB,DUMP,format="LAMMPSDUMP")
    # one  = u.select_atoms("name 1")

    # ave_rdf_1_1_md = InterRDF(one,one,range=(0,5), exclusion_block=(1,1))
    # ave_rdf_1_1_md.run(start=-1, stop=None, step=1, verbose=True)
    
    if mf != '0.0' and mf != '1.0':
        ave_rdf_1_1 = [row[2] for row in ave_block]
        if correction==1:
            ave_rdf_1_1 = ave_rdf_1_1/np.mean(ave_rdf_1_1[1199:1200])
        ave_rdf_1_2 = [row[4] for row in ave_block]
        if correction==1:
            ave_rdf_1_2 = ave_rdf_1_2/np.mean(ave_rdf_1_2[1199:1200])
        ave_rdf_2_2 = [row[6] for row in ave_block]
        if correction==1:
            ave_rdf_2_2 = ave_rdf_2_2/np.mean(ave_rdf_2_2[1199:1200])
            
    if mf == '1.0':
        ave_rdf_1_1 = [row[2] for row in ave_block]
        if correction==1:
            ave_rdf_1_1 = ave_rdf_1_1/np.mean(ave_rdf_1_1[1199:1200])
            
    if mf == '0.0':
        ave_rdf_2_2 = [row[2] for row in ave_block]
        if correction==1:
            ave_rdf_2_2 = ave_rdf_2_2/np.mean(ave_rdf_2_2[1199:1200])

    # Convert the array to a NumPy array
    numpy_array = np.array(blocks)
    # Extract the second element from each row
    second_elements = numpy_array[:, :, 3]
    # Calculate the standard deviation of the second elements along the axis 0
    standard_deviations = np.std(second_elements, axis=0)
    xvals = xvals - np.min(xvals)
    x_fit = x_fit - np.min(x_fit)
    print(mf)
    if mf == str(x1[1]):
         #axes[0,0].plot(ave_rdf_1_1_md.bins, ave_rdf_1_1_md.rdf,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[0,0].plot(xvals,ave_rdf_1_1,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[0,1].plot(xvals,ave_rdf_1_2,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[0,2].plot(xvals,ave_rdf_1_2,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')
    if mf == str(x1[4]):
         axes[3,0].plot(xvals,ave_rdf_1_1,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[3,1].plot(xvals,ave_rdf_1_2,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[3,2].plot(xvals,ave_rdf_1_2,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
    if mf == str(x1[7]):
         axes[6,0].plot(xvals,ave_rdf_1_1,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[6,1].plot(xvals,ave_rdf_1_2,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[6,2].plot(xvals,ave_rdf_1_2,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')
    
    if mf != '0.0' and mf != '1.0':
        
        ave_rdf_1_1 = np.array(ave_rdf_1_1)
        ave_rdf_1_2 = np.array(ave_rdf_1_2)
        ave_rdf_2_2 = np.array(ave_rdf_2_2)
       
        # ave_rdf_1_1 = ave_rdf_1_1[xvals!=0]
        # ave_rdf_1_2 = ave_rdf_1_2[xvals!=0]
        # ave_rdf_2_2 = ave_rdf_2_2[xvals!=0]

        # xvals = xvals[xvals!=0]
            
        dr = xvals[1]-xvals[0]
        KBI_1_1 = 4*np.pi*np.cumsum((ave_rdf_1_1 - 1)*xvals**2*dr )
        KBI_1_2 = 4*np.pi*np.cumsum((ave_rdf_1_2 - 1)*xvals**2*dr )
        KBI_2_2 = 4*np.pi*np.cumsum((ave_rdf_2_2 - 1)*xvals**2*dr )
        
        KBI_1_1_final = 4*np.pi*np.trapz((ave_rdf_1_1 - 1)*xvals**2, dx=dr)
        KBI_1_2_final = 4*np.pi*np.trapz((ave_rdf_1_2 - 1)*xvals**2, dx=dr)
        KBI_2_2_final = 4*np.pi*np.trapz((ave_rdf_2_2 - 1)*xvals**2, dx=dr)
        
        # size dependent KBI, we are assuming an spherical shape rather than cubic (Esteban)
        KBI_1_1_sphere = KBI_Kruger(xvals,ave_rdf_1_1)
        KBI_1_2_sphere = KBI_Kruger(xvals,ave_rdf_1_2)
        KBI_2_2_sphere = KBI_Kruger(xvals,ave_rdf_2_2) 
        
        # sigma_ij = min(xvals[ave_rdf_1_1 !=0])

        #x_ij, y_ij = 1/(8* sigma_ij), 1/(4* sigma_ij)
        
        x_ij, y_ij = 1/5, 1/3
        
        
        KBI_1_1_corrected,KBI_1_1_corrected_error = KBI_extrapolation(xvals,KBI_1_1_sphere,x_ij,y_ij)
        KBI_1_2_corrected,KBI_1_2_corrected_error = KBI_extrapolation(xvals,KBI_1_2_sphere,x_ij,y_ij)
        KBI_2_2_corrected,KBI_2_2_corrected_error = KBI_extrapolation(xvals,KBI_2_2_sphere,x_ij,y_ij)

    if mf == '1.0':
        ave_rdf_1_1 = np.array(ave_rdf_1_1)
    
        dr = xvals[1]-xvals[0]
        KBI_1_1 = 4*np.pi*np.cumsum((ave_rdf_1_1 - 1)*xvals**2*dr )
        
        KBI_1_1_final = 4*np.pi*np.trapz((ave_rdf_1_1 - 1)*xvals**2, dx=dr)
        
        # size dependent KBI, we are assuming an spherical shape rather than cubic (Esteban)
        KBI_1_1_sphere = KBI_Kruger(xvals,ave_rdf_1_1)
        
        # sigma_ij = min(xvals[ave_rdf_1_1 !=0])

        #x_ij, y_ij = 1/(8* sigma_ij), 1/(4* sigma_ij)
        
        x_ij, y_ij = 1/5, 1/3
           
        KBI_1_1_corrected,KBI_1_1_corrected_error = KBI_extrapolation(xvals,KBI_1_1_sphere,x_ij,y_ij)

    if mf == '0.0':
        ave_rdf_2_2 = np.array(ave_rdf_2_2)
    
        dr = xvals[1]-xvals[0]
        KBI_2_2 = 4*np.pi*np.cumsum((ave_rdf_2_2 - 1)*xvals**2*dr )

        KBI_2_2_final = 4*np.pi*np.trapz((ave_rdf_2_2 - 1)*xvals**2, dx=dr)
        
        # size dependent KBI, we are assuming an spherical shape rather than cubic (Esteban)
        KBI_2_2_sphere = KBI_Kruger(xvals,ave_rdf_2_2) 
        
        # sigma_ij = min(xvals[ave_rdf_1_1 !=0])

        #x_ij, y_ij = 1/(8* sigma_ij), 1/(4* sigma_ij)
        
        x_ij, y_ij = 1/5, 1/3
        
        KBI_2_2_corrected,KBI_2_2_corrected_error = KBI_extrapolation(xvals,KBI_2_2_sphere,x_ij,y_ij)
        
    if mf == str(x1[1]):
         axes[1,0].plot(xvals,KBI_1_1,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[1,1].plot(xvals,KBI_1_2,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[1,2].plot(xvals,KBI_2_2,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')
   
         axes[2,0].plot(1/xvals,KBI_1_1_sphere,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[2,1].plot(1/xvals,KBI_1_2_sphere,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[2,2].plot(1/xvals,KBI_2_2_sphere,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')

    if mf == str(x1[3]):
         axes[4,0].plot(xvals,KBI_1_1,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[4,1].plot(xvals,KBI_1_2,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[4,2].plot(xvals,KBI_2_2,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')

         axes[5,0].plot(1/xvals,KBI_1_1_sphere,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[5,1].plot(1/xvals,KBI_1_2_sphere,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[5,2].plot(1/xvals,KBI_2_2_sphere,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')

    if mf == str(x1[7]):
         axes[7,0].plot(xvals,KBI_1_1,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[7,1].plot(xvals,KBI_1_2,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[7,2].plot(xvals,KBI_2_2,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')
         
         axes[8,0].plot(1/xvals,KBI_1_1_sphere,'o',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
         axes[8,1].plot(1/xvals,KBI_1_2_sphere,'o',color=colors[1], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-2')
         axes[8,2].plot(1/xvals,KBI_2_2_sphere,'o',color=colors[2], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 2-2')
    
    axes[0,1].set_yticks([])
    axes[0,2].set_yticks([])
    axes[1,1].set_yticks([])
    axes[1,2].set_yticks([])
    axes[2,1].set_yticks([])
    axes[2,2].set_yticks([])
    axes[3,1].set_yticks([])
    axes[3,2].set_yticks([])
    axes[4,1].set_yticks([])
    axes[4,2].set_yticks([])
    axes[5,1].set_yticks([])
    axes[5,2].set_yticks([])
    axes[6,1].set_yticks([])
    axes[6,2].set_yticks([])
    axes[7,1].set_yticks([])
    axes[7,2].set_yticks([])
    axes[8,1].set_yticks([])
    axes[8,2].set_yticks([])
    
    axes[0,0].set_xticks([])
    axes[0,1].set_xticks([])
    axes[0,2].set_xticks([])
    axes[1,0].set_xticks([])
    axes[1,1].set_xticks([])
    axes[1,2].set_xticks([])
    axes[3,0].set_xticks([])
    axes[3,1].set_xticks([])
    axes[3,2].set_xticks([])
    axes[4,0].set_xticks([])
    axes[4,1].set_xticks([])
    axes[4,2].set_xticks([])
    axes[5,0].set_xticks([])
    axes[5,1].set_xticks([])
    axes[5,2].set_xticks([])
    axes[6,0].set_xticks([])
    axes[6,1].set_xticks([])
    axes[6,2].set_xticks([])
    axes[7,0].set_xticks([])
    axes[7,1].set_xticks([])
    axes[7,2].set_xticks([])
    
    axes[0,0].set_xlim(0,1.05*max(xvals))
    axes[0,1].set_xlim(0,1.05*max(xvals))
    axes[0,2].set_xlim(0,1.05*max(xvals))
    axes[1,0].set_xlim(0,1.05*max(xvals))
    axes[1,1].set_xlim(0,1.05*max(xvals))
    axes[1,2].set_xlim(0,1.05*max(xvals))
    axes[2,0].set_xlim(0,1.05*max(xvals))
    axes[2,1].set_xlim(0,1.05*max(xvals))
    axes[2,2].set_xlim(0,1.05*max(xvals))
    axes[3,0].set_xlim(0,1.05*max(xvals))
    axes[3,1].set_xlim(0,1.05*max(xvals))
    axes[3,2].set_xlim(0,1.05*max(xvals))
    axes[4,0].set_xlim(0,1.05*max(xvals))
    axes[4,1].set_xlim(0,1.05*max(xvals))
    axes[4,2].set_xlim(0,1.05*max(xvals))
    axes[5,0].set_xlim(0,1.05*max(xvals))
    axes[5,1].set_xlim(0,1.05*max(xvals))
    axes[5,2].set_xlim(0,1.05*max(xvals))    
    axes[6,0].set_xlim(0,1.05*max(xvals))
    axes[6,1].set_xlim(0,1.05*max(xvals))
    axes[6,2].set_xlim(0,1.05*max(xvals))   
    axes[7,0].set_xlim(0,1.05*max(xvals))
    axes[7,1].set_xlim(0,1.05*max(xvals))
    axes[7,2].set_xlim(0,1.05*max(xvals))   
    axes[8,0].set_xlim(0,1.05*max(xvals))
    axes[8,1].set_xlim(0,1.05*max(xvals))
    axes[8,2].set_xlim(0,1.05*max(xvals))   
    
    axes[0,0].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[0,1].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[0,2].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[3,0].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[3,1].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[3,2].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[6,0].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[6,1].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)
    axes[6,2].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='24', loc='upper right', handlelength=0.4)

    # axes[3,1].set_xlabel(r'$\mathbf{\textit{r}\ (\sigma)}$',fontsize=40)
    # axes[0,0].set_ylabel(r'$\textbf{RDF}$',fontsize=40)
    # axes[2,0].set_ylabel(r'$\textbf{RDF}$',fontsize=40)
    # axes[4,0].set_ylabel(r'$\textbf{RDF}$',fontsize=40)
    axes[0,0].set_ylabel(r'$\mathbf{\textit{$g_{ij}(r)$}}$', labelpad=45, fontsize=40)
    axes[3,0].set_ylabel(r'$\mathbf{\textit{$g_{ij}(r)$}}$', labelpad=45, fontsize=40)
    axes[6,0].set_ylabel(r'$\mathbf{\textit{$g_{ij}(r)$}}$', labelpad=45, fontsize=40)
    axes[0,0].set_xticks(np.arange(0, max(xvals)*1.1, 40))
    
    # axes[1,0].set_ylabel(r'$\textbf{KBI}$',fontsize=40)
    # axes[3,0].set_ylabel(r'$\textbf{KBI}$',fontsize=40)
    # axes[5,0].set_ylabel(r'$\textbf{KBI}$',fontsize=40)
    axes[1,0].set_ylabel(r'$\mathbf{\textit{$G_{ij}(r)$}}$',fontsize=40)
    axes[4,0].set_ylabel(r'$\mathbf{\textit{$G_{ij}(r)$}}$',fontsize=40)
    axes[7,0].set_ylabel(r'$\mathbf{\textit{$G_{ij}(r)$}}$',fontsize=40)
    axes[2,0].set_ylabel(r'$\mathbf{\textit{$G_{ij}^R(r)$}}$',fontsize=40)
    axes[5,0].set_ylabel(r'$\mathbf{\textit{$G_{ij}^R(r)$}}$',fontsize=40)
    axes[8,0].set_ylabel(r'$\mathbf{\textit{$G_{ij}^R(r)$}}$',fontsize=40)
    
    axes[8,0].set_xlabel(r'$\mathbf{\textit{r}\ (\sigma)}$',fontsize=40)
    axes[8,1].set_xlabel(r'$\mathbf{\textit{r}\ (\sigma)}$',fontsize=40)
    axes[8,2].set_xlabel(r'$\mathbf{\textit{r}\ (\sigma)}$',fontsize=40)

    if mf != '0.0' and mf != '1.0':
        # KBI_1_1_all = np.append(KBI_1_1_all, KBI_1_1[-1])
        # KBI_1_2_all = np.append(KBI_1_2_all, KBI_1_2[-1])
        # KBI_2_2_all = np.append(KBI_2_2_all, KBI_2_2[-1])
        
        # KBI_1_1_all = np.append(KBI_1_1_all, KBI_1_1_final)
        # KBI_1_2_all = np.append(KBI_1_2_all, KBI_1_2_final)
        # KBI_2_2_all = np.append(KBI_2_2_all, KBI_2_2_final)

        KBI_1_1_all = np.append(KBI_1_1_all, KBI_1_1_final)
        KBI_1_2_all = np.append(KBI_1_2_all, KBI_1_2_final)
        KBI_2_2_all = np.append(KBI_2_2_all, KBI_2_2_final)
        
        KBI_1_1_all_corrected = np.append(KBI_1_1_all_corrected, KBI_1_1_corrected)
        KBI_1_2_all_corrected = np.append(KBI_1_2_all_corrected, KBI_1_2_corrected)
        KBI_2_2_all_corrected = np.append(KBI_2_2_all_corrected, KBI_2_2_corrected)

        KBI_1_1_all_corrected_error = np.append(KBI_1_1_all_corrected_error, KBI_1_1_corrected_error)
        KBI_1_2_all_corrected_error = np.append(KBI_1_2_all_corrected_error, KBI_1_2_corrected_error)
        KBI_2_2_all_corrected_error = np.append(KBI_2_2_all_corrected_error, KBI_2_2_corrected_error)
        
    if mf == '1.0':
        KBI_1_1_all = np.append(KBI_1_1_all, KBI_1_1_final)
        KBI_1_1_all_corrected = np.append(KBI_1_1_all_corrected, KBI_1_1_corrected)
        KBI_1_1_all_corrected_error = np.append(KBI_1_1_all_corrected_error, KBI_1_1_corrected_error)

    if mf == '0.0':
        KBI_2_2_all = np.append(KBI_2_2_all, KBI_2_2_final)
        KBI_2_2_all_corrected = np.append(KBI_2_2_all_corrected, KBI_2_2_corrected)
        KBI_2_2_all_corrected_error = np.append(KBI_2_2_all_corrected_error, KBI_2_2_corrected_error)

    if mf != '0.0' and mf != '1.0':
        if correction==1:
            axes[1,0].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[1,1].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[1,2].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[4,0].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[4,1].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[4,2].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[7,0].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[7,1].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
            axes[7,2].set_ylim(-max(KBI_1_1)/0.3,max(KBI_1_1)*2.5)
        if correction==0:
            axes[1,0].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[1,1].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[1,2].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[4,0].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[4,1].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[4,2].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[7,0].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[7,1].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)
            axes[7,2].set_ylim(-max(KBI_1_1)*15.7,max(KBI_1_1)*15.7)

    
plt.show()
pp1.savefig(fig1, bbox_inches='tight')
pp1.close()

correction = 0
if correction==1:
    pp = PdfPages('KBI-corrected_A.pdf')
if correction==0:
    pp = PdfPages('KBI_A.pdf')
    
#fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
# plt.subplots_adjust(wspace=0.0,hspace=0.0)

x2 = [1-value for value in x1]
KBI_1_1_all = [value for value in KBI_1_1_all]
KBI_1_2_all = [value for value in KBI_1_2_all]
KBI_2_2_all = [value for value in KBI_2_2_all]

KBI_1_1_all_corrected = [value for value in KBI_1_1_all_corrected]
KBI_1_2_all_corrected = [value for value in KBI_1_2_all_corrected]
KBI_2_2_all_corrected = [value for value in KBI_2_2_all_corrected]

KBI_1_1_all_corrected_error = [value for value in KBI_1_1_all_corrected_error]
KBI_1_2_all_corrected_error = [value for value in KBI_1_2_all_corrected_error]
KBI_2_2_all_corrected_error = [value for value in KBI_2_2_all_corrected_error]

axes[0,0].plot(x2[1:len(x1)], KBI_1_1_all,'o-',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
axes[0,1].plot(x2[1:len(x1)-1], KBI_1_2_all,'o-',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
axes[0,2].plot(x2[0:len(x1)-1], KBI_2_2_all,'o-',color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')

axes[1,0].errorbar(x2[1:len(x1)], KBI_1_1_all_corrected, yerr=KBI_1_1_all_corrected_error, fmt='o-', color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
axes[1,1].errorbar(x2[1:len(x1)-1], KBI_1_2_all_corrected, yerr=KBI_1_2_all_corrected_error, fmt='o-', color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')
axes[1,2].errorbar(x2[0:len(x1)-1], KBI_2_2_all_corrected, yerr=KBI_2_2_all_corrected_error, fmt='o-', color=colors[0], markersize=8, fillstyle='none', label='x$_{1}$ = '+str(mf_label)+'; 1-1')

axes[0,0].set_ylim(min(KBI_2_2_all)*1.2,-max(KBI_2_2_all)/2.)
axes[0,1].set_ylim(min(KBI_2_2_all)*1.2,-max(KBI_2_2_all)/2.)
axes[0,2].set_ylim(min(KBI_2_2_all)*1.2,-max(KBI_2_2_all)/2.)

axes[1,0].set_ylim(min(KBI_2_2_all)*1.2,-max(KBI_2_2_all)/2.)
axes[1,1].set_ylim(min(KBI_2_2_all)*1.2,-max(KBI_2_2_all)/2.)
axes[1,2].set_ylim(min(KBI_2_2_all)*1.2,-max(KBI_2_2_all)/2.)

axes[0,0].set_ylabel(r'$\mathbf{\mathit{G_{11}^{r_{\mathrm{max}}}}}$', fontsize=40)
axes[0,1].set_ylabel(r'$\mathbf{\mathit{G_{12}^{r_{\mathrm{max}}}}}$', fontsize=40)
axes[0,2].set_ylabel(r'$\mathbf{\mathit{G_{22}^{r_{\mathrm{max}}}}}$', fontsize=40)
axes[0,0].set_xlabel(r'$\mathbf{\textit{x$_{2}$}}$',fontsize=40)
axes[0,1].set_xlabel(r'$\mathbf{\textit{x$_{2}$}}$',fontsize=40)
axes[0,2].set_xlabel(r'$\mathbf{\textit{x$_{2}$}}$',fontsize=40)

axes[1,0].set_ylabel(r'$\mathbf{\textit{G$_{11}^{V}$}}$',fontsize=40)
axes[1,1].set_ylabel(r'$\mathbf{\textit{G$_{12}^{V}$}}$',fontsize=40)
axes[1,2].set_ylabel(r'$\mathbf{\textit{G$_{22}^{V}$}}$',fontsize=40)
axes[1,0].set_xlabel(r'$\mathbf{\textit{x$_{2}$}}$',fontsize=40)
axes[1,1].set_xlabel(r'$\mathbf{\textit{x$_{2}$}}$',fontsize=40)
axes[1,2].set_xlabel(r'$\mathbf{\textit{x$_{2}$}}$',fontsize=40)

plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()
