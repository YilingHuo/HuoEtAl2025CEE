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
from sklearn.decomposition import PCA
import pymannkendall as mk
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
# dir='/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.pl.t.19582014.1000-300hPa/'
os.chdir(dir)
files=sorted(glob.glob("e5.pl.t.*.nc"))
ds=nc.Dataset(files[0])
lat=ds['latitude'][:];level=ds['level'][:];nlat=len(lat);nlev=len(level)

t_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.pl.t_areaavg_lat70.'+str(stryr)+str(endyr)+season+'_combined.dat')
aht_areaavg=np.loadtxt('/global/homes/h/huoyilin/e5.sfc.aht_areaavg_lat70.'+str(stryr)+str(endyr)+season+'.dat')
# Load CO2 data
df = pd.read_csv('/global/cfs/cdirs/m1199/huoyilin/cscratch/co2_mm_mlo.txt', comment='#', delimiter='\s+',
                   names=['year','month','decimal_date','monthly_avg',
                          'de_seasonalized','days','std','uncertainty'])
co2_avgln=np.log(seasonavg(df.monthly_avg[(df.year>=stryr1) & (df.year<=endyr)],m1,m2,nyear)/278)
# co2_avgln/=np.std(co2_avgln)
df = pd.read_csv('/global/cfs/cdirs/m1199/huoyilin/cscratch/monthly.ao.index.b50.current.ascii.table', comment='#', delimiter='\s+',
                   names=['year','Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'])
ao_avg=(((df.Jan[(df.year>=stryr) & (df.year<=endyr)]).to_numpy()+(df.Feb[(df.year>=stryr) & (df.year<=endyr)]).to_numpy()+(df.Dec[(df.year>=stryr1) & (df.year<=endyr-1)])).to_numpy())/3;del df

####start with the AR/dynamically induced and AR/dynamically adjusted SIC termn then the CO2 regression
# ar_areaavg=ao_avg ###use something to replace ARs
ar_areaavg=aht_areaavg ###use something to replace ARs
clev=0.01#confidence level
Detrend=False ###whether the regression coefficients are based on detrended data
if Detrend:    
    ar_areaavg0=signal.detrend(ar_areaavg);ar_areaavg0-=np.mean(ar_areaavg0)
    sic_areaavg0=signal.detrend(sic_areaavg);sic_areaavg0-=np.mean(sic_areaavg0)
    co2_avgln0=signal.detrend(co2_avgln)
else:    
    ar_areaavg0=ar_areaavg+0;sic_areaavg0=sic_areaavg+0;co2_avgln0=co2_avgln+0
t_areaavg0=t_areaavg+0
nfactor=3
indexyr=np.arange(stryr,endyr+1)/10
res_ar=np.zeros(nlev)+np.nan;res_sic=np.zeros(nlev)+np.nan;res_co2=np.zeros(nlev)+np.nan
t_trend=np.polyfit(indexyr,t_areaavg,1)[-2]
t_reconst_trend=np.zeros(nlev)+np.nan
t_trend_contribution=np.zeros([nfactor,nlev])+np.nan
corr_obsreg=np.zeros([4,nlev])+np.nan##double the size to 2*2 for p value
residual_sic=sic_areaavg-LinearRegression().fit(ar_areaavg0.reshape(-1, 1), sic_areaavg0).coef_*ar_areaavg
for ilev in range(nlev):
    result= mk.original_test(t_areaavg[:,ilev]);    t_trend[ilev]=result.slope*10
    print('t_trend.p of level '+str(ilev)+' < '+str(clev)+': '+str(result.p<clev))
    if Detrend:    t_areaavg0[:,ilev]=signal.detrend(t_areaavg[:,ilev]);
    res_ar[ilev]=smOLSconfidencelev(ar_areaavg0,t_areaavg0[:,ilev],clev)[0]
    residual=t_areaavg0[:,ilev]-res_ar[ilev]*ar_areaavg0
    res_sic[ilev]=smOLSconfidencelev(residual_sic,residual,clev)[0]
    residual-=res_sic[ilev]*residual_sic
    res_co2[ilev]=smOLSconfidencelev(co2_avgln0,residual,clev)[0]
    t_reconst=res_ar[ilev]*ar_areaavg+res_sic[ilev]*residual_sic+res_co2[ilev]*co2_avgln
    my_model = PCA(n_components=nfactor)
    my_model.fit_transform([res_ar[ilev]*ar_areaavg, res_sic[ilev]*residual_sic, res_co2[ilev]*co2_avgln])
    t_trend_contribution[0,ilev]=explained_variance_score(t_reconst, res_ar[ilev]*ar_areaavg)
    t_trend_contribution[-2,ilev]=explained_variance_score(t_reconst, res_sic[ilev]*residual_sic)
    t_trend_contribution[-1,ilev]=explained_variance_score(t_reconst, res_co2[ilev]*co2_avgln)
    t_trend_contribution[:,ilev]/=np.sum(t_trend_contribution[:,ilev])
    res = stats.linregress(t_areaavg[:,ilev],t_reconst)
    corr_obsreg[0,ilev]=res.rvalue
    corr_obsreg[2,ilev]=res.pvalue
    res = stats.linregress(signal.detrend(t_areaavg[:,ilev]),signal.detrend(t_reconst))
    corr_obsreg[1,ilev]=res.rvalue
    corr_obsreg[3,ilev]=res.pvalue
    result= mk.original_test(t_reconst);    t_reconst_trend[ilev]=result.slope*10
    print('t_reconst_trend.p of level '+str(ilev)+' < '+str(clev)+': '+str(result.p<clev))
if Detrend:t_reconst_trend0=t_reconst_trend+0
fntsz=18
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 10), facecolor='w', gridspec_kw={'width_ratios': [1, 1.7]})
ax1 = axes[0]
ax1.scatter(t_reconst_trend, level[-nlev:],s=200,c='r', alpha=1.0,label='Regressed')#*t_trend_contribution[0]
ax1.scatter(t_trend, level[-nlev:],s=100,c='k',label='Observed')
ax1.set_ylim(max(level[-nlev:])+10, min(level[-nlev:])-10)
ax1.set_ylim(max(level[-nlev:])+10, 375)
ax1.set_title('(a)',loc='left', fontsize=fntsz)
ax1.set_xlabel('T trend (K decade$^{-1}$)', fontsize=fntsz);plt.xlim(0, 1.1)
ax1.set_ylabel('Pressure (hPa)', fontsize=fntsz)
ax1.legend(frameon=False,labelcolor='linecolor', fontsize=fntsz+5,markerscale=0)
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5);
# plot bars
left = nlev * [0]
factors=['PET','SIC',r'$CO_{2}$']
colors = 'kbr'
labels = [factors[0],factors[0]+'-adjusted '+factors[1],r'$CO_{2}$']
ax1 = axes[1]
for ifactor in range(nfactor):
    ax1.barh(np.arange(0,nlev), t_trend_contribution[ifactor]*100, left = left, color=colors[ifactor])
    left += t_trend_contribution[ifactor]*100
ax1.set_xlim(0, 100)
ax1.set_ylim(nlev-.5,-.5)
ax1.set_yticks(np.arange(0,nlev),level[-nlev:].astype(int))
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.tick_params(axis='both', which='minor',length=5);
# title, legend, labels
ax1.set_title('(b)',loc='left', fontsize=fntsz)
ax1.legend(labels,bbox_to_anchor=([.15, 1.08, 0, 0]),  frameon=False, fontsize=fntsz+5,labelcolor='linecolor', handletextpad=0.0, handlelength=0,loc='upper left',ncol=nfactor)#bbox_to_anchor=([1.02, 1, 0, 0])
ax1.set_xlabel('Contribution to regressed T trends (%)', fontsize=fntsz);ax1.set_ylabel('Pressure (hPa)', fontsize=fntsz)
plt.show()
fn='/pscratch/sd/h/huoyilin/Fig2.nc'
if os.path.exists(fn):        os.remove(fn)
ncfile = nc.Dataset(fn,mode='w',format='NETCDF4_CLASSIC') 
pressure_level_dim = ncfile.createDimension('pressure_level', None) # unlimited axis (can be appended to).
ncfile.comment='Data for Huo et al. (2025)'
pressure_level = ncfile.createVariable('pressure_level', np.float64, ('pressure_level',))
pressure_level.units = 'millibars'
pressure_level.long_name = 'pressure_level'
pressure_level[:]=level
# Define a 2D variable to hold the data
t_trend_obs = ncfile.createVariable('t_trend_obs',np.float64,('pressure_level')) # note: unlimited dimension is leftmost
t_trend_obs.long_name = 'Observed temperature trends' # this is a CF standard name
t_trend_obs.units='K decade-1'
t_trend_obs[:]=t_trend
t_trend_reg = ncfile.createVariable('t_trend_reg',np.float64,('pressure_level')) # note: unlimited dimension is leftmost
t_trend_reg.long_name = 'Regressed temperature trends' # this is a CF standard name
t_trend_reg[:]=t_reconst_trend
t_trend_reg.units='K decade-1'
PET_contribution = ncfile.createVariable('PET_contribution',np.float64,('pressure_level')) # note: unlimited dimension is leftmost
PET_contribution.long_name = 'Contribution of PET to regressed T trends' # this is a CF standard name
PET_contribution[:]=t_trend_contribution[0]
PET_contribution.units='1'
SIC_contribution = ncfile.createVariable('SIC_contribution',np.float64,('pressure_level')) # note: unlimited dimension is leftmost
SIC_contribution.long_name = 'Contribution of PET-adjusted SIC to regressed T trends' # this is a CF standard name
SIC_contribution[:]=t_trend_contribution[1]
SIC_contribution.units='1'
CO2_contribution = ncfile.createVariable('CO2_contribution',np.float64,('pressure_level')) # note: unlimited dimension is leftmost
CO2_contribution.long_name = 'Contribution of CO2 to regressed T trends' # this is a CF standard name
CO2_contribution[:]=t_trend_contribution[2]
CO2_contribution.units='1'
ncfile.close()
