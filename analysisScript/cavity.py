from slab import *
from guiqwt.pyplot import *
from matplotlib.pyplot import *
from os.path import join
from glob import glob
set_fit_plotting(pkg='matplotlib')

filepath="S:\\_Data\\120926 - Three Cavities\\"
analysis="Analysis_Ge\\"
datafolder="120930_PowerSweep_000\\"
plotfolder="plots"
plotpath=os.path.join(filepath,analysis,plotfolder)

#os.mkdir(os.path.join(filepath,analysis))
#os.mkdir(os.path.join(filepath,analysis,plotfolder)) 

centers1=[6.9413897e9,10.1033e9,11.052202e9,12.85992e9,13.246393e9,13.2690308e9,13.300962e9]
centers3=[7.4189334e9,10.068677e9]
centers4=[6.1184369e9,9.0939845e9,12.591482e9]
centers5=[12.628142e9]
centers=centers1+centers3+centers4+centers5
#centers=[centers1,centers3,centers4,centers5]
powers=['0.000000', '-5.000000', '-10.000000', '-15.000000', '-20.000000', '-25.000000', '-30.000000', 
        '-35.000000', '-40.000000', '-45.000000', '-50.000000', '-55.000000', '-60.000000', 
        '-65.000000', '-70.000000', '-75.000000', '-80.000000','-85.000000']
        
powers=arange(0,-90,-5)
center=centers[0]
power = powers[0]

def sprlot(center,power,offset=1.0):
    fpath=glob(join(filepath,datafolder)+str(power)+"*"+str(center/1e9)[0:6]+"*.CSV")[0]
    data=load_nwa_file(fpath)+power*offset
    #subplot(4,1,1)
    plot(data[0]/1.e9,data[1],label=str(power)+' dBm')
    xlim(data[0][0]/1.e9,data[0][-1]/1.e9)
    
def psplot(center,offset=1.0,powers=powers):
    for power in powers:
        sprlot(center,power,offset=10.0)
    xlabel('Frequency (f in GHz)')
    ylabel('S21 (Log Scale)')
    
def cavplot(centers):
    figsize(10*len(centers),8);subplot(len(centers),2,1);subplot(13,2,1)
    for count in range(0,len(centers)):
        subplot(1,len(centers),count+1)
        title('Cavity with center frequency '+str(centers[count]/1e9)[0:6]+'GHz')
        psplot(centers[count],10.0)
        #legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ylabel('S21 (Log Scale) with an offset for each')
        legend()

##Fitting Resonance Data
        
def spload(center,power):
    fpath=glob(join(filepath,datafolder)+str(power)+"*"+str(center/1e9)[0:4]+"*.CSV")[0]
    return load_nwa_file(fpath)
def sploads(center,powers):
    data=[]
    for power in powers:
        data.append(spload(center,power))
    return data  
    
    
from slab.dsfit import fitlor
def pswit(d,center,width=None):
    if width!=None:
        domain=(center/1.e6-width,center/1.e6+width)
    else : domain = None
    ylabel('Transmission (linear Scale)');xlabel('Freq (f in MHz)')
    fit=fitlor(d[0]/1e6,dBmtoW(d[1]),showfit=True,domain=domain)
    xlim(d[0][0]/1.e6,d[0][-1]/1.e6)
    #ylim(-1e-8,5e-8);#print((d[0][0]/1.e6,d[0][-1]/1.e6))
    return fit
    
def powit (center,width): 
    figsize(3*5.,2*len(powers)/5.);count=0;fits=[]
    data=sploads(center,powers)
    ymax=max(dBmtoW(data[0][1]))*1.5
    for d in data:
        count=count+1
        subplot(len(data)/5+1,5,count)
        fits.append(pswit(d,center,width))
        ylim(0,ymax)
    return transpose(fits)