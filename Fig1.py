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
stryr=1980;endyr=2024;nyear=endyr-stryr+1
ptop=400
m1=11;m2=2
stryr1=stryr+0
seasons='JFMAMJJASOND'
if m2-m1>11:
    season='Annual'
elif m2<m1:
    season=seasons[m1:]+seasons[:m2]
    stryr1-=1
else:
    season=seasons[m1:m2]
dir='/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.pl.t.'+str(stryr1)+'2024.1000-'+str(ptop)+'hPa/'
os.chdir(dir)
files=sorted(glob.glob("e5.pl.t.*.nc"))
ds=nc.Dataset(files[0])
lat=ds['latitude'][:];level=ds['level'][:];nlat=len(lat);nlev=len(level)
t=np.zeros([(endyr-stryr1+1)*12,nlev])
for fn in files:
    year1=int(fn[8:12]);year2=int(fn[12:16])
    if not (year2<stryr1 or year1>endyr):
        ds=nc.Dataset(fn)
        if 'level' in ds.dimensions:        
            level1=ds['level'][:]
        else:
            level1=ds['pressure_level'][:]
        nlev1=sum(level1>=ptop)
        # ERA5 = xr.open_mfdataset(fn,combine='by_coords') ## open the data
        # ERA5_combine =ERA5.sel(expver=1).combine_first(ERA5.sel(expver=5))
        # ERA5.load()
        if level1[0]<level1[1]:
            if year1>=stryr1:    t[12*(year1-stryr1):12*(year2-stryr1+1)]=areaavg_lat(ds['t'][:,-nlev1:],lat)
            else: t[:12*(year2-stryr1+1)]=areaavg_lat(ds['t'][12*(stryr1-year1):,-nlev1:],lat)
        else:
            if year1>=stryr1:    t[12*(year1-stryr1):12*(year2-stryr1+1)]=np.flip(areaavg_lat(ds['t'][:,:nlev1],lat),axis=1)
            else: t[:12*(year2-stryr1+1)]=np.flip(areaavg_lat(ds['t'][12*(stryr1-year1):,:nlev1],lat),axis=1)            
        ds.close()
        del level1,nlev1,ds #ERA5

t_areaavg=seasonavg(t,m1,m2,nyear);del t
for ilev in range(nlev):
    t_areaavg[:,ilev]-=np.mean(t_areaavg[:,ilev])
np.savetxt('/global/homes/h/huoyilin/e5.pl.t_areaavg_lat70.'+str(stryr)+str(endyr)+season+'_combined.dat', t_areaavg) ###for faster access next time
t_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.pl.t_areaavg_lat70.'+str(stryr)+str(endyr)+season+'_combined.dat')
fn='/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.avg_slhtf_snlwrf_snlwrfcs_snswrf_snswrfcs_ishf_tnlwrf_tnlwrfcs_tnswrf_tnswrfcs.19502024.nc'
ds1=nc.Dataset(fn);lat=ds1['latitude'][:];year1=int(fn[-11:-7])
aht=(-ds1['avg_tnswrf'][:]+ds1['avg_snswrf'][:]-ds1['avg_tnlwrf'][:]+ds1['avg_snlwrf'][:]+ds1['avg_slhtf'][:]+ds1['avg_ishf'][:])[12*(stryr1-year1):12*(endyr-year1+1)]#-333550*(ds2['mtpr'][12*(stryr-filestryr):12*(endyr-filestryr+1)]))
aht_areaavg=areaavg_lat(seasonavg(aht,m1,m2,nyear),lat);del aht
aht_areaavg-=np.mean(aht_areaavg)# ar_areaavg/=np.std(ar_areaavg)
np.savetxt('/global/homes/h/huoyilin/e5.sfc.aht_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat', aht_areaavg)###for faster access next time
aht_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.sfc.aht_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat')
fn='/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.vign_vithen_viten_viwvn_vipile.19502024.nc'
ds1=nc.Dataset(fn);lat=ds1['latitude'][:];year1=int(fn[-11:-7])
tcwvf=ds1['viwvn'][(stryr1-year1)*12:(endyr-year1+1)*12]
tcwvf_areaavg=areaavg_lat(seasonavg(tcwvf,m1,m2,nyear),lat);del tcwvf
tcwvf_areaavg-=np.mean(tcwvf_areaavg)
np.savetxt('/global/homes/h/huoyilin/e5.sfc.tcwvf_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat', tcwvf_areaavg)###for faster access next time
tcwvf_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.sfc.tcwvf_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat')

fn = '/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.t2m_msl_sp_siconc_tcw_tcwv.19402024.nc'
ds=nc.Dataset(fn);lat=ds['latitude'][:]
ERA5 = xr.open_mfdataset(fn,combine='by_coords') ## open the data
ERA5.load();ERA5_combine=ERA5
year1=int(fn[-11:-7])
sic=ERA5_combine['siconc'][(stryr11-year1)*12:(endyr1-year1+1)*12]
sic_areaavg1=areaavg_lat(seasonavg(sic,m11,m21,nyear),lat)
sic_areaavg1-=np.mean(sic_areaavg1)
sic=ERA5_combine['siconc'][(stryr1-year1)*12:(endyr-year1+1)*12]
sic_areaavg=areaavg_lat(seasonavg(sic,m1,m2,nyear),lat);del sic
sic_areaavg-=np.mean(sic_areaavg)
# sic_areaavg/=np.std(sic_areaavg)
np.savetxt('/global/homes/h/huoyilin/e5.sfc.siconc_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat', sic_areaavg)###for faster access next time
sic_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.sfc.siconc_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat')
t2m=ERA5_combine['t2m'][(stryr1-year1)*12:(endyr-year1+1)*12]
t2m_areaavg=areaavg_lat(seasonavg(t2m,m1,m2,nyear),lat);del t2m
t2m_areaavg-=np.mean(t2m_areaavg)
np.savetxt('/global/homes/h/huoyilin/e5.sfc.t2m_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat', t2m_areaavg)###for faster access next time
t2m_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.sfc.t2m_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat')
# Load CO2 data
df = pd.read_csv('/global/cfs/cdirs/m1199/huoyilin/cscratch/co2_mm_mlo.txt', comment='#', delimiter='\s+',
                   names=['year','month','decimal_date','monthly_avg',
                          'de_seasonalized','days','std','uncertainty'])
co2_avgln=np.log(seasonavg(df.monthly_avg[(df.year>=stryr1) & (df.year<=endyr)],m1,m2,nyear)/278)
df = pd.read_csv('/global/cfs/cdirs/m1199/huoyilin/cscratch/monthly.ao.index.b50.current.ascii.table', comment='#', delimiter='\s+',
                   names=['year','Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'])
ao_avg=(((df.Jan[(df.year>=stryr) & (df.year<=endyr)]).to_numpy()+(df.Feb[(df.year>=stryr) & (df.year<=endyr)]).to_numpy()+(df.Dec[(df.year>=stryr1) & (df.year<=endyr-1)])).to_numpy())/3;del df
switched=False
t2m_areaavg/=np.nanstd(t2m_areaavg);sic_areaavg/=np.nanstd(sic_areaavg);tcwvf_areaavg/=np.nanstd(tcwvf_areaavg);aht_areaavg/=np.nanstd(aht_areaavg)#ar_areaavg/=np.nanstd(ar_areaavg);
fntsz=15;linew=3
indexyr=np.arange(stryr,endyr+1)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 5), facecolor='w')
labels=['SAT','-SIC','ARs','PMT','PET']
ax1 = axes[0]
ax1.plot(indexyr, t2m_areaavg,c='r',lw=linew, label=labels[0])#*t_trend_contribution[0]
ax1.plot(indexyr, -sic_areaavg,c='b',lw=linew, label=labels[1])#*t_trend_contribution[0]
ax1.plot(indexyr, aht_areaavg,c='k',lw=linew, label=labels[-1])#*t_trend_contribution[0] poleward moisture transport
ax1.set_ylim(-3, 3);ax1.set_xlim(stryr, endyr)
ax1.set_title('(a)',loc='left', fontsize=fntsz)
ax1.set_ylabel('Standardized anomalies', fontsize=fntsz)
ax1.legend(frameon=False,labelcolor='linecolor', fontsize=fntsz,handlelength=0)
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5)
ax1.text(indexyr[0]+.5, -2.9, 'R('+labels[0]+', '+labels[1]+') = '+"%.2f (p = %.3f)\n" %(np.corrcoef(t2m_areaavg, -sic_areaavg)[0,1],stats.linregress(t2m_areaavg, -sic_areaavg).pvalue)\
         +'R('+labels[0]+', '+labels[-1]+') = '+"%.2f (p = %.3f)\n" %(np.corrcoef(t2m_areaavg, aht_areaavg)[0,1],stats.linregress(t2m_areaavg, aht_areaavg).pvalue)\
         +'R('+labels[1]+', '+labels[-1]+') = '+"%.2f (p = %.3f)" %(np.corrcoef(-sic_areaavg, aht_areaavg)[0,1],stats.linregress(-sic_areaavg, aht_areaavg).pvalue)\
         ,fontsize=fntsz)
ax1.axhline(y = 0, color = 'k', alpha = .5,lw=linew/3)
ax1 = axes[1]
t2m_areaavg_detrend=signal.detrend(t2m_areaavg);sic_areaavg_detrend=signal.detrend(sic_areaavg);tcwvf_areaavg_detrend=signal.detrend(tcwvf_areaavg);aht_areaavg_detrend=signal.detrend(aht_areaavg)#ar_areaavg_detrend=signal.detrend(ar_areaavg);
ax1.plot(indexyr, t2m_areaavg_detrend,c='r',lw=linew)#*t_trend_contribution[0]
ax1.plot(indexyr, -sic_areaavg_detrend,c='b',lw=linew)#*t_trend_contribution[0]
ax1.plot(indexyr, aht_areaavg_detrend,c='k',lw=linew)#*t_trend_contribution[0] poleward moisture transport
ax1.set_ylim(-3, 3);ax1.set_xlim(stryr, endyr)
ax1.set_title('(b)',loc='left', fontsize=fntsz)
ax1.set_ylabel('Standardized detrended anomalies', fontsize=fntsz)
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5);
ax1.text(indexyr[0]+.5, -2.9, 'R('+labels[0]+', '+labels[1]+') = '+"%.2f (p = %.3f)\n" %(np.corrcoef(t2m_areaavg_detrend, -sic_areaavg_detrend)[0,1],stats.linregress(t2m_areaavg_detrend, -sic_areaavg_detrend).pvalue)\
         +'R('+labels[0]+', '+labels[-1]+') = '+"%.2f (p = %.3f)\n" %(np.corrcoef(t2m_areaavg_detrend, aht_areaavg_detrend)[0,1],stats.linregress(t2m_areaavg_detrend, aht_areaavg_detrend).pvalue)\
         +'R('+labels[1]+', '+labels[-1]+') = '+"%.2f (p = %.3f)" %(np.corrcoef(-sic_areaavg_detrend, aht_areaavg_detrend)[0,1],stats.linregress(-sic_areaavg_detrend, aht_areaavg_detrend).pvalue)\
         ,fontsize=fntsz)
ax1.axhline(y = 0, color = 'k', alpha = .5,lw=linew/3)
plt.show()
ncfile = nc.Dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/Fig1.nc',mode='w',format='NETCDF4_CLASSIC') 
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
ncfile.comment='Data for Huo et al. (2025)'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'years since 1979'
# time.long_name = 'time'
time[:]=indexyr-stryr+1
# Define a 3D variable to hold the data
SAT = ncfile.createVariable('SAT',np.float64,('time')) # note: unlimited dimension is leftmost
SAT.long_name = 'Standardized anomalies of SAT' # this is a CF standard name
SAT[:]=t2m_areaavg
SIC = ncfile.createVariable('SIC',np.float64,('time')) # note: unlimited dimension is leftmost
SIC.long_name = 'Standardized anomalies of SIC' # this is a CF standard name
SIC[:]=sic_areaavg
PET = ncfile.createVariable('PET',np.float64,('time')) # note: unlimited dimension is leftmost
PET.long_name = 'Standardized anomalies of PET' # this is a CF standard name
PET[:]=aht_areaavg
SAT_detrended = ncfile.createVariable('SAT_detrended',np.float64,('time')) # note: unlimited dimension is leftmost
SAT_detrended.long_name = 'Standardized detrended anomalies of SAT' # this is a CF standard name
SAT_detrended[:]=t2m_areaavg_detrend
SIC_detrended = ncfile.createVariable('SIC_detrended',np.float64,('time')) # note: unlimited dimension is leftmost
SIC_detrended.long_name = 'Standardized detrended anomalies of SIC' # this is a CF standard name
SIC_detrended[:]=sic_areaavg_detrend
PET_detrended = ncfile.createVariable('PET_detrended',np.float64,('time')) # note: unlimited dimension is leftmost
PET_detrended.long_name = 'Standardized detrended anomalies of PET' # this is a CF standard name
PET_detrended[:]=aht_areaavg_detrend
ncfile.close()
