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


exp_name = ['cntrl_2wks','stddev10_2wks'] 
file = [' ',' ']
print('exp_name=',exp_name) 
print('len(exp_name)=',len(exp_name))
year = 2019
month = 6
day_list = [14,15,16,17,18, 19,20,21,22,23,24,25,26,27]
hour_list = [0, 6, 12, 18]

ncyc=len(day_list)*len(hour_list)
nexp = len(exp_name)
mean_omf_all = np.zeros([nexp,ncyc])
std_omf_all = np.zeros([nexp,ncyc])
mean_oma_all = np.zeros([nexp,ncyc])
std_oma_all = np.zeros([nexp,ncyc])
mean_omf_all_npp = np.zeros([nexp,ncyc])
std_omf_all_npp = np.zeros([nexp,ncyc])
mean_oma_all_npp = np.zeros([nexp,ncyc])
std_oma_all_npp = np.zeros([nexp,ncyc])
cyc_num = np.zeros([nexp,ncyc])
iexp=0
while iexp < len(exp_name):
   if exp_name[iexp] == 'cntrl_2wks':
      qc_flag = 13
   else:
      qc_flag = 0
   icyc=0
   for day in day_list:
      for hour in hour_list:
         yyyymmddhh_str = str(year)+str(month).zfill(2)+str(day).zfill(2)+str(hour).zfill(2)
         data_path = '/scratch1/NCEPDEV/stmp2/Andrew.Tangborn/ROTDIR/'+exp_name[iexp]+'/gdas.'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'/'+str(hour).zfill(2)+'/chem/scratch1/NCEPDEV/stmp2/Andrew.Tangborn/RUNDIRS/'+exp_name[iexp]+'/'+str(year)+str(month).zfill(2)+str(day).zfill(2)+str(hour).zfill(2)+'/gdas/aeroanl_'+str(year)+str(month).zfill(2)+str(day).zfill(2)+str(hour).zfill(2)+'/diags/'
         filename = 'diag_viirs_n20_'+yyyymmddhh_str+'_0000.nc4'
         filename_npp = 'diag_viirs_npp_'+yyyymmddhh_str+'_0000.nc4'
         print('filename=',filename)
         fn = Path(data_path+filename)
         fn_npp = Path(data_path+filename) 
         print('fn=',fn) 
         if fn.exists():
             print('fn=',fn)
             datain = nc.Dataset(fn,'r') 
             datain_npp = nc.Dataset(fn_npp,'r')
             meta_data = datain.groups['MetaData']
             bkgmob_group = datain.groups['bkgmob']
             anlmob_group= datain.groups['anlmob']
             qc_group = datain.groups['EffectiveQC0']
             bkgmob_group_npp = datain_npp.groups['bkgmob']
             anlmob_group_npp = datain_npp.groups['anlmob']
             qc_group_npp = datain_npp.groups['EffectiveQC0']
             omf1 = bkgmob_group.variables['aerosol_optical_depth']
             oma1 = anlmob_group.variables['aerosol_optical_depth']
             qc1 = qc_group.variables['aerosol_optical_depth']
             qc = np.squeeze(qc1,axis=1)
             omf = np.squeeze(omf1,axis=1)
             oma = np.squeeze(oma1,axis=1)
             print('qc_shape=',qc.shape)
             print('omf_shape=',omf.shape)
             omf = omf[qc==qc_flag]
             oma = oma[qc==qc_flag]
             omf_npp = bkgmob_group_npp.variables['aerosol_optical_depth']
             oma_npp = anlmob_group_npp.variables['aerosol_optical_depth']
             qc1_npp = qc_group_npp.variables['aerosol_optical_depth']
             qc_npp = np.squeeze(qc1_npp,axis=1)
             omf_npp = omf_npp[qc==qc_flag]
             oma_npp = oma_npp[qc==qc_flag]
             mean_omf = np.nanmean(omf)
             mean_oma = np.nanmean(oma)
             std_omf = np.nanstd(omf)
             std_oma = np.nanstd(oma)
             mean_omf_npp = np.nanmean(omf_npp)
             mean_oma_npp = np.nanmean(oma_npp)
             std_omf_npp = np.nanstd(omf_npp)
             std_oma_npp = np.nanstd(oma_npp)

             mean_omf_all[iexp,icyc] = mean_omf
             mean_oma_all[iexp,icyc] = mean_oma
             std_omf_all[iexp,icyc] = std_omf
             std_oma_all[iexp,icyc] = std_oma
             mean_omf_all_npp[iexp,icyc] = mean_omf_npp
             mean_oma_all_npp[iexp,icyc] = mean_oma_npp
             std_omf_all_npp[iexp,icyc] = std_omf_npp
             std_oma_all_npp[iexp,icyc] = std_oma_npp
             print(iexp,icyc,'mean_omf_all[iexp,icyc]= ',mean_omf_all[iexp,icyc])
        
         icyc+=1 
   iexp+=1


const_zero = np.zeros(icyc)
print('length(mean_omf_all)=',len(mean_omf_all[0,:]))
plt.figure(0)
plt.plot( mean_omf_all[0,0:len(mean_omf_all[0,:]-2)],"-b",label="Mean OmF "+exp_name[0],linewidth=1.0)
#plt.plot( mean_oma_all[0,0:len(mean_oma_all[0,:]-2)],"-r",label="Mean OmA "+exp_name[0],linewidth=1.0)
plt.plot( std_omf_all[0,0:len(std_omf_all[0,:]-2)],"-.b",label="Std OmF "+exp_name[0], linewidth=1.0)
#plt.plot(std_oma_all[0,0:len(std_omf_all[0,:]-2)],"-.r",label="Std OmA "+exp_name[0], linewidth=1.0)
plt.plot( mean_omf_all[1,0:len(mean_omf_all[1,:]-2)],"-g",label="Mean OmF "+exp_name[1],linewidth=1.0)
#plt.plot( mean_oma_all[1,0:len(mean_oma_all[1,:]-2)],"-m",label="Mean OmA "+exp_name[1],linewidth=1.0)
plt.plot( std_omf_all[1,0:len(std_omf_all[1,:]-2)],"-.g",label="Std OmF "+exp_name[1], linewidth=1.0)
#plt.plot(std_oma_all[1,0:len(std_omf_all[1,:]-2)],"-.m",label="Std OmA "+exp_name[1], linewidth=1.0)
plt.plot( const_zero[:],"-.k",linewidth=.5)
plt.xlabel("Cycle")
plt.ylabel("Mean (Std)")# OmF(OmA)")
plt.legend(loc="lower left", prop={'size':6})
plt.savefig('omf_stats_'+exp_name[0]+'_'+exp_name[1]+'.png')

plt.figure(1)
plt.plot( mean_omf_all_npp[0,0:len(mean_omf_all[0,:]-2)],"-b",label="Mean OmF "+exp_name[0],linewidth=1.0)
#plt.plot( mean_oma_all_npp[0,0:len(mean_oma_all[0,:]-2)],"-r",label="Mean OmA "+exp_name[0],linewidth=1.0)
plt.plot( std_omf_all_npp[0,0:len(std_omf_all[0,:]-2)],"-.b",label="Std OmF "+exp_name[0], linewidth=1.0)
#plt.plot(std_oma_all_npp[0,0:len(std_omf_all[0,:]-2)],"-.r",label="Std OmA "+exp_name[0], linewidth=1.0)
plt.plot( mean_omf_all_npp[1,0:len(mean_omf_all[1,:]-2)],"-g",label="Mean OmF "+exp_name[1],linewidth=1.0)
#plt.plot( mean_oma_all_npp[1,0:len(mean_oma_all[1,:]-2)],"-m",label="Mean OmA "+exp_name[1],linewidth=1.0)
plt.plot( std_omf_all_npp[1,0:len(std_omf_all[1,:]-2)],"-.g",label="Std OmF "+exp_name[1], linewidth=1.0)
#plt.plot(std_oma_all_npp[1,0:len(std_omf_all[1,:]-2)],"-.m",label="Std OmA "+exp_name[1], linewidth=1.0)
plt.plot( const_zero[:],"-.k",linewidth=.5)
plt.xlabel("Cycle")
plt.ylabel("Mean (Std)")# OmF(OmA)")
plt.legend(loc="lower left", prop={'size':6})
plt.savefig('omf_stats_'+exp_name[0]+'_'+exp_name[1]+'_npp.png')

