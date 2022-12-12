import numpy as np
import os 
from pathlib import Path
import math
import matplotlib.pyplot as plt
import subprocess as sp
#import matplotlib.pyplot as plt
#from emcpy.plots import CreateMap
#from emcpy.plots.map_tools import Domain, MapProjection
#from emcpy.plots.map_plots import MapGridded
import netCDF4 as nc
import glob
import array
import tarfile 
import time 

datapath = '/scratch1/NCEPDEV/stmp2/Andrew.Tangborn/'
exp_dir = 'ROTDIR'
exp_name = 'stddev50_2wks'
year = 2019
month = 6
day_list = [14,15,16,17,18, 19,20,21,22,23,24,25, 26, 27] 
hour_list = [0, 6, 12, 18]
#hour = 6
ncyc=len(day_list)*len(hour_list)-1 
mean_omf_all = np.zeros(ncyc)
std_omf_all = np.zeros(ncyc)
mean_oma_all = np.zeros(ncyc)
std_oma_all = np.zeros(ncyc)
mean_omf_all_npp = np.zeros(ncyc)
std_omf_all_npp = np.zeros(ncyc)
mean_oma_all_npp = np.zeros(ncyc)
std_oma_all_npp = np.zeros(ncyc)
cyc_num = np.zeros(ncyc) 

my_env = os.environ.copy()
my_env['OMP_NUM_THREADS'] = '4' # for openmp to speed up fortran call

icyc=0
mean_omf = []
mean_oma = [] 
std_omf = []
std_oma = [] 

for day in day_list: 
   for hour in hour_list:
      print('hour = ', hour) 
      chem_path = datapath+exp_dir+'/'+exp_name+'/gdas.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'/'+str(hour).zfill(2)+'/chem/'
      tardir = datapath+exp_dir+'/'+exp_name+'/gdas.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'/'+str(hour).zfill(2)+'/chem/'
      print('chem_path= ',chem_path)
      print('tardir = ', tardir)
      tar_file = datapath+exp_dir+'/'+exp_name+'/gdas.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'/'+str(hour).zfill(2)+'/chem/gdas.t'+str(hour).zfill(2)+'z.aerostat'
      print('tar_file=',tar_file)
# Check if tar file exists
      myfile_tar = Path(tar_file)
      if myfile_tar.exists():
# Determine path to tar diag file  
         cmd = 'tar -tf '+tar_file
         print(' cmd = ', cmd) 
         proc = sp.Popen(cmd,env=my_env,shell=True,stdout=sp.PIPE)
         output = proc.stdout.read()
         len1 = int(len(output)/2)
         output1 = output[0:len1]
         output_npp = output[len1:len(output)] 
         print('output_npp = ', output_npp)
         output1a = output1.strip()
         output_npp_a = output_npp.strip()
         print('output_npp_a =', output_npp_a)
         print('output1a=',output1a)
         my_env = os.environ.copy()
         my_env['OMP_NUM_THREADS'] = '4' 
         cmd2 = 'tar -xvf '+tar_file+' -C ' +tardir
         print('cmd2 = ', cmd2) 
         proc2 = sp.Popen(cmd2,env=my_env,shell=True,stdout=sp.PIPE) 
         output2 = proc2.stdout.read()
         print('output2=',output2)
#   os.system(cmd2) 
#   os.system('ls')
# Path and filename of O-F stats file. 
         fn=chem_path+output1a.decode("utf-8")
         fn_npp = chem_path+output_npp_a.decode("utf-8") 
         print('output1a= ',output1a)
         
         print('fn= ',fn)
         print('fn_npp= ',fn_npp)
         myfile = Path(fn) 
         if myfile.exists():
            print(fn,' exists')
            cmd3 = 'gunzip '+fn 
            cmd4 = 'gunzip '+fn_npp
            sp.Popen(cmd3,env=my_env,shell=True)
            sp.Popen(cmd4,env=my_env,shell=True)
         fn_nc_tmp = fn[0:len(fn)-3]
         fn_nc = str(fn_nc_tmp)
         fn_npp_tmp = fn_npp[0:len(fn_npp)-3]
         fn_npp = str(fn_npp_tmp) 
         res = isinstance(fn_nc, str)
         print('fn_nc = ', fn_nc) 
         if res:
           print('fn_nc_1=',fn_nc)
         else:
           fn_nc = fn[0:len(fn)-3].decode("utf-8")
           fn_npp = fn_npp[0:len(fn_npp)-3].decode("utf-8")
           print('fn_nc_2=',fn_nc) 
   
         print('fn_nc=',fn_nc) 
         ncfile = Path(fn_nc)
         if ncfile.exists():
            datain = nc.Dataset(fn_nc,'r')
            datain_npp = nc.Dataset(fn_npp,'r')
         else:
            time.sleep(20)
            datain = nc.Dataset(fn_nc,'r')
            datain_npp = nc.Dataset(fn_npp,'r')

         meta_data = datain.groups['MetaData']
         meta_data_npp = datain.groups['MetaData']
         print('meta_data=',meta_data)

         bkgmob_group = datain.groups['bkgmob']
         anlmob_group = datain.groups['anlmob']
         qc_group = datain.groups['EffectiveQC0']
        
         bkgmob_group_npp = datain_npp.groups['bkgmob']
         anlmob_group_npp = datain_npp.groups['anlmob']
         qc_group_npp = datain_npp.groups['EffectiveQC0']

         omf = bkgmob_group.variables['aerosol_optical_depth']
         oma = anlmob_group.variables['aerosol_optical_depth']
         qc1 = qc_group.variables['aerosol_optical_depth']
         qc = np.squeeze(qc1,axis=1)
         print('qc=',qc) 
         index_qc1 = omf[qc==0] 
         index_qc=np.squeeze(index_qc1,axis=1)
         print('shape(qc1)=',qc1.shape)
         print('shape(qc)=',qc.shape)
         print('shape(index_qc)=',index_qc.shape)
         print('shape(index_qc1)=',index_qc1.shape)
         omf_qc = omf[qc==0]
         oma_qc = oma[qc==0]
         omf_npp = bkgmob_group_npp.variables['aerosol_optical_depth']
         oma_npp = anlmob_group_npp.variables['aerosol_optical_depth']
         qc1_npp = qc_group_npp.variables['aerosol_optical_depth']
         qc_npp = np.squeeze(qc1_npp,axis=1)
         index_qc_npp = np.where(qc_npp==0) 
         omf_qc_npp = omf_npp[index_qc_npp]
         oma_qc_npp = oma_npp[index_qc_npp]
         print('icyc=',icyc)
         mean_omf = np.nanmean(omf)
         mean_omf_qc = np.nanmean(omf_qc)
         mean_oma = np.nanmean(oma) 
         mean_oma_qc = np.nanmean(oma_qc)
         std_omf = np.nanstd(omf)
         std_omf_qc = np.nanstd(omf_qc)
         std_oma = np.nanstd(oma) 
         std_oma_qc = np.nanstd(oma_qc)
         mean_omf_all[icyc] = mean_omf_qc
         mean_oma_all[icyc] = mean_oma_qc
         std_omf_all[icyc] = std_omf_qc
         std_oma_all[icyc] = std_oma_qc

         mean_omf_npp = np.nanmean(omf_npp)
         mean_oma_npp = np.nanmean(oma_npp)
         mean_omf_qc_npp = np.nanmean(omf_qc_npp)
         mean_oma_qc_npp = np.nanmean(oma_qc_npp)
         std_omf_npp = np.nanstd(omf_npp)
         std_oma_npp = np.nanstd(oma_npp)
         std_omf_qc_npp = np.nanstd(omf_qc_npp)
         std_oma_qc_npp = np.nanstd(oma_qc_npp)
         mean_omf_all_npp[icyc] = mean_omf_qc_npp
         mean_oma_all_npp[icyc] = mean_oma_qc_npp
         std_omf_all_npp[icyc] = std_omf_qc_npp
         std_oma_all_npp[icyc] = std_oma_qc_npp


         print('mean_omf = ', mean_omf)
         print('mean_oma = ', mean_oma) 
         print('std_omf = ', std_omf)
         print('std_oma = ', std_oma) 
         cyc_num [icyc] = icyc
         icyc = icyc + 1 

print('mean_omf_all=',mean_omf_all)
print('mean_oma_all=',mean_oma_all)
const_zero = np.zeros(icyc)
print('omf=',omf) 
print('size(mean_omf=',mean_omf_all.shape)
print('size(const_zero=',const_zero.shape)
plt.figure(0)
plt.plot( mean_omf_all[0:len(mean_omf_all-2)],"-b",label="Mean OmF",linewidth=1.0)
plt.plot( mean_oma_all[0:len(mean_oma_all-2)],"-r",label="Mean OmA",linewidth=1.0)
plt.plot( std_omf_all[0:len(std_omf_all-2)],"-.b",label="Std OmF", linewidth=1.0)
plt.plot(std_oma_all[0:len(std_omf_all-2)],"-.r",label="Std OmA", linewidth=1.0)
plt.plot( const_zero[:],"-.k",linewidth=.5)
plt.xlabel("Cycle")
plt.ylabel("Mean (Std) OmF(OmA)")
plt.legend(loc="lower left")
plt.savefig('mean_omf_oma_vs_cycle.png')


plt.figure(1)
plt.plot( mean_omf_all_npp[0:len(mean_omf_all_npp-2)],"-b",label="Mean OmF",linewidth=1.0)
plt.plot( mean_oma_all_npp[0:len(mean_oma_all_npp-2)],"-r",label="Mean OmA",linewidth=1.0)
plt.plot( std_omf_all_npp[0:len(std_omf_all_npp-2)],"-.b",label="Std OmF", linewidth=1.0)
plt.plot(std_oma_all_npp[0:len(std_omf_all_npp-2)],"-.r",label="Std OmA", linewidth=1.0)
plt.plot( const_zero[:],"-.k",linewidth=.5)
plt.xlabel("Cycle")
plt.ylabel("Mean (Std) OmF(OmA)")
plt.legend(loc="lower left")
plt.savefig('mean_omf_oma_vs_cycle_npp.png')
