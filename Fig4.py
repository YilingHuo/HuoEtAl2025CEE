import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.basemap import Basemap
import pandas as pd
from scipy import stats
import xarray as xr
import pymannkendall as mk
from scipy import signal
import os
import glob
from sklearn.metrics import (explained_variance_score,mean_squared_error,r2_score)
import Ngl
import copy
from sklearn import preprocessing
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
def stddataanomaly(x):
    ntim=len(x)
    x_tavg=np.nanmean(x,axis=0)
    x_std=np.nanstd(x,axis=0)
    x_a=x*1
    for itim in range(ntim):
        x_a[itim]-=x_tavg
        x_a[itim]/=x_std
    return x_a
def areaavg_lat(x,lat,latlim=70):
    nsize=len(x.shape)
    nlatsize=len(lat.shape)
    if nlatsize>1: nlat=len(lat.flatten())
    else: nlat=len(lat)
    if nlatsize<2:
        if lat[1]<lat[0]:lat1 = np.flip(lat,axis=0)
        else: lat1 = lat+0
        latdiff=lat+0;latdiff[1:-1]=(lat1[2:]-lat1[:-2])/2;latdiff[0]=(lat1[1]+lat1[0])/2+90;latdiff[-1]=90-(lat1[-1]+lat1[-2])/2
        if lat[1]<lat[0]: latdiff = np.flip(latdiff,axis=0)
        latweights = np.where(lat>latlim,np.cos(np.deg2rad(lat))*latdiff,0)
    else: latweights= np.where(lat>latlim,np.cos(np.deg2rad(lat)),0)
    if nlatsize==1 and nsize>1 and nlat!=x.shape[-1]:
        latweights =np.transpose(np.vstack([latweights]*x.shape[-1]))
    #Area weight
    if nsize>2 or (nsize>1 and nlat==x.shape[-1]):
        ntim=len(x)
        if nsize>3 or (nsize>2 and nlat==x.shape[-1]):
            nlev=len(x[0]);xareaavg=np.zeros((ntim,nlev))+np.nan
            for itim in range(ntim):
                for ilev in range(nlev):
                    tmp=x[itim,ilev];
                    if np.isnan(tmp).all():xareaavg[itim,ilev]=np.nan
                    else:ii = ~np.isnan(tmp);xareaavg[itim,ilev]=np.average(tmp[ii], weights=latweights[ii])
        else:
            xareaavg=np.zeros((ntim))+np.nan
            for itim in range(ntim):
                tmp=x[itim]
                if np.isnan(tmp).all():xareaavg[itim]=np.nan
                else: ii = ~np.isnan(tmp);xareaavg[itim]=np.average(tmp[ii], weights=latweights[ii])
        return xareaavg        
    else:
        if np.isnan(x).all():return np.nan
        else: ii = ~np.isnan(x); return np.average(x[ii], weights=latweights[ii])
def seasonavg(x,m1,m2,nyear):
    nsize=len(x.shape)
    if nsize>3:
        x_tavg=np.zeros([nyear,len(x[0]),len(x[0][0]),len(x[0][0][0])])
    elif nsize>2:
        x_tavg=np.zeros([nyear,len(x[0]),len(x[0][0])])
    elif nsize>1:
        x_tavg=np.zeros([nyear,len(x[0])])
    elif nsize==1:
        x_tavg=np.zeros([nyear])
    x_tavg+=np.nan
    if m2<m1:
        for iyear in range(nyear):
            x_tavg[iyear]=(np.nansum(x[iyear*12+12:iyear*12+12+m2],axis=0)+np.nansum(x[iyear*12+m1:iyear*12+12],axis=0))/(12+m2-m1)
    else:
        for iyear in range(nyear):
            x_tavg[-iyear-1]=np.nanmean(x[-12-iyear*12+m1:-12-iyear*12+m2],axis=0)
        if m2==12: x_tavg[-1]=np.nanmean(x[-12+m1:],axis=0)            
    return x_tavg
def seasonget(x,m1,m2,nyear):
    nmonth=m2-m1
    if m2<m1: nmonth=12+m2-m1
    x_t=copy.deepcopy(x[:nyear*nmonth])
    if m2<m1:
        for iyear in range(nyear):
            x_t[iyear*nmonth:iyear*nmonth+12-m1]=x[iyear*12+m1:iyear*12+12]
            x_t[iyear*nmonth+12-m1:iyear*nmonth+nmonth]=x[iyear*12+12:iyear*12+12+m2]
    else:
        if m2==12:
            x_t[-nmonth:]=x[-12+m1:]
        else:
            x_t[-nmonth:]=x[-12+m1:-12+m2]           
        for iyear in range(1,nyear):
            x_t[-iyear*nmonth-nmonth:-iyear*nmonth]=x[-12-iyear*12+m1:-12-iyear*12+m2]
    return x_t
def smOLSconfidencelev(x,y,clev):
    res=sm.OLS(y, sm.add_constant(x)).fit()
    parmNconf=np.zeros([2])
    parmNconf[0]=res.params[1]
    parmNconf[1]=res.params[1]-res.conf_int(clev)[1][0]
    return parmNconf
def trenddetector(list_of_index, array_of_data, order=1):
    mask = ~np.isnan(list_of_index) & ~np.isnan(array_of_data)
    return float(np.polyfit(list_of_index[mask], list(array_of_data[mask]), order)[-2])
# Two-sided inverse Students t-distribution
# p - probability, df - degrees of freedom
tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
def statslinregressts_slopeinterval(x,y,clev):
    mask = ~np.isnan(x) & ~np.isnan(y)
    ts= tinv(clev, sum(mask+0)-2)
    res = stats.linregress(x[mask], y[mask])
    return res.slope,ts*res.stderr
def statslinregressts_slopepvalue(x,y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if len(y[mask])<1:   return np.nan,np.nan,np.nan
    else:
        res = stats.linregress(x[mask], y[mask])
        return res.slope,res.pvalue,np.square(res.rvalue)
clev=0.05#confidence level
stryr=1959;endyr=2015;nyear=endyr-stryr+1
m1=11;m2=2
seasons='JFMAMJJASOND'
stryr1=stryr*1
if m2-m1>11:
    season='Annual'
elif m2<m1:
    season=seasons[m1:]+seasons[:m2]
    stryr1-=1 ##for DJF read in one previous year
else:
    season=seasons[m1:m2]

ds=nc.Dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/G10010_SIBT1850_V2/G10010_sibt1850_v2.0.nc');sicnm='seaice_conc';filstryr=1850;filendyr=2017##NSIDC
lat=ds['latitude'][:];lon=ds['longitude'][:]
nlat=len(lat);nlon=len(lon)
if endyr<filendyr+1:
    sic=ds[sicnm][(stryr1-filstryr)*12:(endyr-filstryr+1)*12]##NSIDC
    sic_areaavg=areaavg_lat(seasonavg(sic,m1,m2,nyear),lat);del sic
else:
    adjyr=10;adjstryr=filendyr-adjyr+1##adjust the ERA5 data by subtracting the 10-year mean monthly differences between the two reanalyses
    sic=ds[sicnm][(stryr1-filstryr)*12:(filendyr-filstryr+1)*12]##NSIDC
    sic_areaavg=areaavg_lat(seasonavg(sic,m1,m2,filendyr-stryr+1),lat)
    sic=ds[sicnm][(adjstryr-filstryr)*12:(filendyr-filstryr+1)*12]
    sic_areaavg1=np.mean(areaavg_lat(seasonavg(sic,m1,m2,filendyr-adjstryr+1),lat),axis=0)
    fn = '/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.t2m_msl_sp_siconc_tcw_tcwv.19402024.nc';sicnm='siconc';filstryr=int(fn[-11:-7])
    ERA5_combine = xr.open_mfdataset(fn,combine='by_coords') ## open the data    
    sic=ERA5_combine[sicnm][(adjstryr-filstryr)*12:(endyr-filstryr+1)*12].values;lat=ERA5_combine['latitude'].values;del ERA5_combine
    sic_areaavgera5=areaavg_lat(seasonavg(sic,m1,m2,endyr-adjstryr+1),lat);del sic
    sic_areaavg1-=np.mean(sic_areaavgera5[:adjyr],axis=0)
    sic_areaavg=np.concatenate((sic_areaavg,sic_areaavgera5[adjyr:]+sic_areaavg1), axis=0)
sic_areaavg-=np.mean(sic_areaavg)
ds=nc.Dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc')
lat=ds['latitude'][:];lon=ds['longitude'][:]
nlat=len(lat);nlon=len(lon)
t=ds['tas_mean'][(stryr1-1850)*12:(endyr-1849)*12]
t_areaavg=areaavg_lat(seasonavg(t,m1,m2,nyear),lat)
t_areaavg-=np.mean(t_areaavg)
# Load CO2 data
df = pd.read_csv('/global/cfs/cdirs/m1199/huoyilin/cscratch/co2_mm_mlo.txt', comment='#', delimiter='\s+',
                   names=['year','month','decimal_date','monthly_avg_co2',
                          'de_seasonalized','days','std','uncertainty'])
co2_avg=seasonavg(df.monthly_avg_co2[(df.year>=stryr1) & (df.year<=endyr)],m1,m2,nyear);del df
co2_avgln=np.log(co2_avg/278)
co2_avgln-=np.mean(co2_avgln)
nfactor=3
Detrend=False
if Detrend:    
    t_areaavg=signal.detrend(t_areaavg);t_areaavg-=np.mean(t_areaavg)
    sic_areaavg=signal.detrend(sic_areaavg);sic_areaavg-=np.mean(sic_areaavg)
    co2_avgln=signal.detrend(co2_avgln)
indexyr=np.arange(1,nyear+1)/10
t_trend=statslinregressts_slopeinterval(indexyr,t_areaavg,clev)
folder='/global/homes/h/huoyilin/benb0228-PLS-Dynamical-Adjustment-a1db5bc/'
SLP_contribution=np.loadtxt(folder+'HadCRUT5NSIDC.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand1SLP_contribution'+season+'.dat')
sic_dynadj=sic_areaavg-smOLSconfidencelev(SLP_contribution,sic_areaavg,clev)[0]*SLP_contribution
t_reconst=np.zeros([nfactor+1,len(t_areaavg)])+np.nan
SLP_contribution=np.loadtxt(folder+'HadCRUT5NSIDC.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand0SLP_contribution'+season+'.dat')
t_reconst[1]=statslinregressts_slopeinterval(SLP_contribution,t_areaavg,clev)[0]*SLP_contribution
residual=t_areaavg-t_reconst[1]
res_sic=statslinregressts_slopeinterval(sic_dynadj,residual,clev)[0]
t_reconst[-2]=res_sic*sic_dynadj
residual-=t_reconst[-2]
res_co2=statslinregressts_slopeinterval(co2_avgln,residual,clev)[0]
t_reconst[-1]=res_co2*co2_avgln
t_reconst[0]=np.nansum(t_reconst[1:],axis=0)
t_reconst_trend=np.zeros([nfactor+1,2])+np.nan
for ifactor in range(nfactor+1):
    tmp=t_reconst[ifactor]
    mask = ~np.isnan(t_areaavg) & ~np.isnan(tmp)
    t_reconst_trend[ifactor]=statslinregressts_slopeinterval(indexyr[mask],tmp[mask],clev)
print(mk.yue_wang_modification_test(t_areaavg, alpha=.01).h)
for ifactor in range(nfactor+1):
    print(mk.yue_wang_modification_test(t_reconst[ifactor], alpha=.01).p)
movingwindow=30
nmovingwindow=len(t_areaavg)-movingwindow#30-year moving windows
t_trend_mvw=np.zeros([nfactor+2,nmovingwindow])+np.nan
for imvw in range(nmovingwindow):
    if sum(np.isnan(t_areaavg[imvw:movingwindow+imvw])+0)<movingwindow/2:
        t_trend_mvw[0,imvw]=statslinregressts_slopeinterval(indexyr[0:movingwindow],t_areaavg[imvw:movingwindow+imvw],1)[0]
    else:
        t_trend_mvw[0,imvw]=np.nan
    for ifactor in range(nfactor+1):
        tmp=t_reconst[ifactor]+0
        mask=np.isnan(t_areaavg) | np.isnan(tmp)
        tmp[mask]=np.nan
        if sum(mask[imvw:movingwindow+imvw]+0)<movingwindow/2:
            t_trend_mvw[ifactor+1,imvw]=statslinregressts_slopeinterval(indexyr[0:movingwindow],tmp[imvw:movingwindow+imvw],1)[0]
        else:
            t_trend_mvw[ifactor+1,imvw]=np.nan
fntsz=11.5;linew=2.5;padvalue=-170
indexyr=np.arange(stryr,endyr+1)
ylimt=np.nanmax(abs(np.concatenate((t_areaavg, t_reconst[0]))))+.1;colors='krb';letters='abcdefghijklmn'
v1=-.1;v2=.7
factors=['Dynamically-induced','Dynamically-adjusted SIC',r'$CO_{2}$']
fig, axes = plt.subplots(nrows=nfactor+1, ncols=2,figsize=(12, 12.5), facecolor='w', gridspec_kw={'width_ratios': [5, 1]})
ax1 = axes[0,0]
ax1.plot(indexyr,t_areaavg,color='k',lw=linew,label='Observed, trend = '+"%.2f" %(t_trend[0])+u"\u00B1"+"%.2f" %(t_trend[1]))
ax1.plot(indexyr,t_reconst[0],color='r',lw=linew,label='Regressed, trend = '+"%.2f" %(t_reconst_trend[0][0])+u"\u00B1"+"%.2f" %(t_reconst_trend[0][1]))
ax1.legend(frameon=False, fontsize=fntsz, labelcolor='linecolor', handletextpad=0.0, handlelength=0)
tmp=t_reconst[0]
mask = ~np.isnan(t_areaavg) & ~np.isnan(tmp)
ax1.text(indexyr[0]+1.1, ylimt/6, 'R = '+"%.2f (" %(np.corrcoef(t_areaavg[mask],tmp[mask])[0,1])+"%.2f)\n"%(np.corrcoef(signal.detrend(t_areaavg[mask]),signal.detrend(tmp[mask]))[0,1])+'RMSD = '+"%.2f\n"%mean_squared_error(t_areaavg,t_reconst[0],squared=False),fontsize=fntsz)
ax1.set_ylim(-ylimt, ylimt)
ax1.set_xlim(stryr, endyr)
dd=4
ax1.set_xticks(indexyr[::dd],(str(iyear+dd) for iyear in indexyr[::dd]), fontsize = fntsz)
ax1.xaxis.set_tick_params(labelbottom=False)
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5);
ax1.axhline(y = 0, color = 'k', linestyle = '--',lw=linew/3)
ax1.set_ylabel('SAT anomalies (K)', fontsize = fntsz)
ax1.set_title('(a)', fontsize = fntsz+3, loc='right', y=1.0, pad=padvalue)
ax1 = axes[0,1]
for i in range(2):
    ax1.errorbar(i,np.nanmean(t_trend_mvw[i]),yerr=np.nanstd(t_trend_mvw[i]),color=colors[i],fmt=".", mfc='none',ms=25, capsize=7)
    ax1.plot(i,np.nanmean(t_trend_mvw[i]),'o',ms=3,color=colors[i])
ax1.set_ylim(v1,v2)
ax1.set_xlim(-.5, 1.5)
ax1.set_xticks([0,1],[])
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5);
ax1.set_ylabel('SAT trend (K decade$^{-1}$)', fontsize = fntsz)
ax1.set_title('('+letters[nfactor+1]+')', fontsize = fntsz+3, loc='left', y=1.0, pad=padvalue)
print (explained_variance_score(t_areaavg,t_reconst[0]))
total_explained_variance=explained_variance_score(t_reconst[0],t_reconst[1])+explained_variance_score(t_reconst[0],t_reconst[2])+explained_variance_score(t_reconst[0],t_reconst[3])
print (total_explained_variance)
for ifactor in range(1,nfactor+1):
    ax1 = axes[ifactor,0]
    if mk.yue_wang_modification_test(t_reconst[ifactor], alpha=.01).p >0.01:
        ax1.plot(indexyr,t_reconst[ifactor],color='r',lw=linew,label='trend = '+"%.2f" %(t_reconst_trend[ifactor][0])+u"\u00B1"+"%.2f" %(t_reconst_trend[ifactor][1])+' (p = '+"%.2f" %(mk.yue_wang_modification_test(t_reconst[ifactor]).p)+')')
    else:
        ax1.plot(indexyr,t_reconst[ifactor],color='r',lw=linew,label='trend = '+"%.2f" %(t_reconst_trend[ifactor][0])+u"\u00B1"+"%.2f" %(t_reconst_trend[ifactor][1]))
    ax1.legend(frameon=False, fontsize=fntsz, labelcolor='linecolor', handletextpad=0.0, handlelength=0)
    ax1.set_ylim(-ylimt, ylimt)
    ax1.set_xlim(stryr, endyr)
    ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
    ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
    ax1.tick_params(axis='both', which='minor',length=5);
    ax1.set_xticks(indexyr[::dd], labels=[i for i in indexyr[::dd]])
    if ifactor<nfactor:     ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.axhline(y = 0, color = 'k', linestyle = '--',lw=linew/3)
    ax1.set_ylabel(factors[ifactor-1]+' (K)', fontsize = fntsz)
    ax1.set_title('('+letters[ifactor]+')', fontsize = fntsz+3, loc='right', y=1.0, pad=padvalue)
    ax1 = axes[ifactor,1]
    ax1.errorbar(i,np.nanmean(t_trend_mvw[ifactor+1]),yerr=np.nanstd(t_trend_mvw[ifactor+1]),color='r',fmt=".", mfc='none',ms=25, capsize=7)
    ax1.plot(i,np.nanmean(t_trend_mvw[ifactor+1]),'o',ms=3,color='r')
    ax1.set_ylim(v1,v2)
    ax1.set_xlim(-.5, 1.5)
    if ifactor==nfactor: 
        ax1.set_xticks([0,1],['Observed','Regressed'], fontsize = fntsz)
    else:
        ax1.set_xticks([0,1],[])
    ax1.set_ylabel(factors[ifactor-1]+' (K decade$^{-1}$)', fontsize = fntsz)
    ax1.set_ylabel(factors[ifactor-1]+' (decade$^{-1}$)', fontsize = fntsz)
    ax1.set_title('('+letters[ifactor+nfactor+1]+')', fontsize = fntsz+3, loc='left', y=1.0, pad=padvalue)
    ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
    ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
    ax1.tick_params(axis='y', which='minor',length=5);    ax1.tick_params(axis='x', which='minor',length=0);
    print(explained_variance_score(t_reconst[0],t_reconst[ifactor])/total_explained_variance)
fig.tight_layout()
plt.show()
fn='/pscratch/sd/h/huoyilin/Fig4.nc'
if os.path.exists(fn):        os.remove(fn)
ncfile = nc.Dataset(fn,mode='w',format='NETCDF4_CLASSIC') 
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
ncfile.comment='Data for Huo et al. (2025)'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'years since 1958';time[:]=indexyr-stryr+1
# Define a 2D variable to hold the data
SAT_obs = ncfile.createVariable('SAT_obs',np.float64,('time')) # note: unlimited dimension is leftmost
SAT_obs.long_name = 'Area-averaged DJF-mean SAT variations over the Arctic based on the HadCRUT5 data' # this is a CF standard name
SAT_obs.units='K';SAT_obs[:]=t_areaavg
SAT_reg = ncfile.createVariable('SAT_reg',np.float64,('time')) # note: unlimited dimension is leftmost
SAT_reg.long_name = 'Area-averaged DJF-mean SAT variations over the Arctic reproduced by the stepwise regression' # this is a CF standard name
SAT_reg.units='K';SAT_reg[:]=t_reconst[0]
SAT_dynamically_induced = ncfile.createVariable('SAT_dynamically_induced',np.float64,('time')) # note: unlimited dimension is leftmost
SAT_dynamically_induced.long_name = 'Contributions to the regressed SAT from dynamically induced component' # this is a CF standard name
SAT_dynamically_induced.units='K';SAT_dynamically_induced[:]=t_reconst[1]
SAT_SIC = ncfile.createVariable('SAT_SIC',np.float64,('time')) # note: unlimited dimension is leftmost
SAT_SIC.long_name = 'Contributions to the regressed SAT from dynamically adjusted SIC' # this is a CF standard name
SAT_SIC.units='K';SAT_SIC[:]=t_reconst[2]
SAT_CO2 = ncfile.createVariable('SAT_CO2',np.float64,('time')) # note: unlimited dimension is leftmost
SAT_CO2.long_name = 'Contributions to the regressed SAT from CO2' # this is a CF standard name
SAT_CO2.units='K';SAT_CO2[:]=t_reconst[3]
SAT_trend_obs_mean = ncfile.createVariable('SAT_trend_obs_mean',np.float64) # note: unlimited dimension is leftmost
SAT_trend_obs_mean.long_name = 'Mean of SAT trends from 30-year moving windows based on observed anomaly time series' # this is a CF standard name
SAT_trend_obs_mean.units='K per decade';SAT_trend_obs_mean[:]=np.nanmean(t_trend_mvw[0])
SAT_trend_obs_std = ncfile.createVariable('SAT_trend_obs_std',np.float64) # note: unlimited dimension is leftmost
SAT_trend_obs_std.long_name = 'Standard deviation of SAT trends from 30-year moving windows based on observed anomaly time series' # this is a CF standard name
SAT_trend_obs_std.units='K per decade';SAT_trend_obs_std[:]=np.nanstd(t_trend_mvw[0])
SAT_trend_reg_mean = ncfile.createVariable('SAT_trend_reg_mean',np.float64) # note: unlimited dimension is leftmost
SAT_trend_reg_mean.long_name = 'Mean of SAT trends from 30-year moving windows based on regressed anomaly time series' # this is a CF standard name
SAT_trend_reg_mean.units='K per decade';SAT_trend_reg_mean[:]=np.nanmean(t_trend_mvw[1])
SAT_trend_reg_std = ncfile.createVariable('SAT_trend_reg_std',np.float64) # note: unlimited dimension is leftmost
SAT_trend_reg_std.long_name = 'Standard deviation of SAT trends from 30-year moving windows based on regressed anomaly time series' # this is a CF standard name
SAT_trend_reg_std.units='K per decade';SAT_trend_reg_std[:]=np.nanstd(t_trend_mvw[1])
SAT_trend_dynamically_induced_mean = ncfile.createVariable('SAT_trend_dynamically_induced_mean',np.float64) # note: unlimited dimension is leftmost
SAT_trend_dynamically_induced_mean.long_name = 'Mean of SAT trends from 30-year moving windows based on contributions to the regressed SAT from dynamically induced component' # this is a CF standard name
SAT_trend_dynamically_induced_mean.units='K per decade';SAT_trend_dynamically_induced_mean[:]=np.nanmean(t_trend_mvw[2])
SAT_trend_dynamically_induced_std = ncfile.createVariable('SAT_trend_dynamically_induced_std',np.float64) # note: unlimited dimension is leftmost
SAT_trend_dynamically_induced_std.long_name = 'Standard deviation of SAT trends from 30-year moving windows based on observed anomaly time series' # this is a CF standard name
SAT_trend_dynamically_induced_std.units='K per decade';SAT_trend_dynamically_induced_std[:]=np.nanstd(t_trend_mvw[2])
SAT_trend_SIC_mean = ncfile.createVariable('SAT_trend_SIC_mean',np.float64) # note: unlimited dimension is leftmost
SAT_trend_SIC_mean.long_name = 'Mean of SAT trends from 30-year moving windows based on contributions to the regressed SAT from dynamically adjusted SIC' # this is a CF standard name
SAT_trend_SIC_mean.units='K per decade';SAT_trend_SIC_mean[:]=np.nanmean(t_trend_mvw[3])
SAT_trend_SIC_std = ncfile.createVariable('SAT_trend_SIC_std',np.float64) # note: unlimited dimension is leftmost
SAT_trend_SIC_std.long_name = 'Standard deviation of SAT trends from 30-year moving windows based on contributions to the regressed SAT from dynamically adjusted SIC' # this is a CF standard name
SAT_trend_SIC_std.units='K per decade';SAT_trend_SIC_std[:]=np.nanstd(t_trend_mvw[3])
SAT_trend_CO2_mean = ncfile.createVariable('SAT_trend_CO2_mean',np.float64) # note: unlimited dimension is leftmost
SAT_trend_CO2_mean.long_name = 'Mean of SAT trends from 30-year moving windows based on contributions to the regressed SAT from CO2' # this is a CF standard name
SAT_trend_CO2_mean.units='K per decade';SAT_trend_CO2_mean[:]=np.nanmean(t_trend_mvw[4])
SAT_trend_CO2_std = ncfile.createVariable('SAT_trend_CO2_std',np.float64) # note: unlimited dimension is leftmost
SAT_trend_CO2_std.long_name = 'Standard deviation of SAT trends from 30-year moving windows based on contributions to the regressed SAT from CO2' # this is a CF standard name
SAT_trend_CO2_std.units='K per decade';SAT_trend_CO2_std[:]=np.nanstd(t_trend_mvw[4])
ncfile.close()
