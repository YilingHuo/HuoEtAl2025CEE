import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import glob

from scipy import stats
import pymannkendall as mk
from bisect import bisect_right

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from sklearn.cross_decomposition import PLSRegression   
from sklearn.linear_model import LinearRegression
from scipy import signal
import statsmodels.api as sm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import math
# reduce warnings (less pink)
import warnings
warnings.filterwarnings('ignore')
import os
def detrend_X(X, x_for_plots, num_years = 50, Detrend_or_Filter = 'Detrend'):
    if Detrend_or_Filter == 'None':
        X_for_corr_map = X
        pass    
    elif Detrend_or_Filter == 'Linear_Detrend':
        model = LinearRegression()
        # Detrend X
        X_for_corr_map = np.zeros((X.shape[0],X.shape[1]))
        temp_X_axis = x_for_plots.reshape(len(x_for_plots),1)
        for grid_cell in range(X.shape[1]):
            temp_X = X[:,grid_cell]
            model.fit(temp_X_axis, temp_X)
            X_for_corr_map[:,grid_cell] = temp_X - model.predict(temp_X_axis)
    elif Detrend_or_Filter == 'Diff_Detrend':
        # detrending leads to (Years - 1) values
        # detrend by differencing: https://machinelearningmastery.com/time-series-trends-in-python/
        # Detrend X
        X_for_corr_map = np.zeros((X.shape[0]-1,X.shape[1]))
        for grid_cell in range(X.shape[1]):
            X_temp_grid_cell = []
            for yr in range(1, X.shape[0]):
                value = X[yr,grid_cell] - X[yr - 1,grid_cell]
                X_temp_grid_cell.append(value)
            X_for_corr_map[:,grid_cell] = X_temp_grid_cell
    elif Detrend_or_Filter == 'Filter':
        # set-up filter with 15-yr high pass
        sampleRate = 15 / num_years
        Nyquist_frequency = sampleRate/2
        sos = signal.butter(N=1, Wn=Nyquist_frequency, btype='highpass',output='sos')

        # Filter X
        X_for_corr_map = np.zeros((X.shape[0],X.shape[1]))
        for grid_cell in range(X.shape[1]):
            X_grid_cell_series = X[:,grid_cell]
            X_for_corr_map[:,grid_cell] = signal.sosfilt(sos, X_grid_cell_series)
    return X_for_corr_map

def detrend_Y(Y, x_for_plots, num_years = 50, Detrend_or_Filter = 'Detrend'):
    if Detrend_or_Filter == 'None':
        Y_for_corr_map = Y
    elif Detrend_or_Filter == 'Linear_Detrend':
        model = LinearRegression()        
        # Detrend Y
        model.fit(x_for_plots.reshape(len(x_for_plots),1), Y)
        Y_for_corr_map = Y - model.predict(x_for_plots.reshape(len(x_for_plots),1))        
    elif Detrend_or_Filter == 'Diff_Detrend':
        # detrending leads to (Years - 1) values
        # detrend by differencing: https://machinelearningmastery.com/time-series-trends-in-python/
        # Detrend Y
        Y_for_corr_map_list = []
        for yr in range(1, len(Y)):
            value = Y[yr] - Y[yr - 1]
            Y_for_corr_map_list.append(value)
        Y_for_corr_map = np.array(Y_for_corr_map_list)
        #Y_for_corr_map = Y_for_corr_map.reshape(Y_for_corr_map.shape[0])        
    elif Detrend_or_Filter == 'Filter':
        # set-up filter with 15-yr high pass
        sampleRate = 15 / num_years
        Nyquist_frequency = sampleRate/2
        sos = signal.butter(N=1, Wn=Nyquist_frequency, btype='highpass',output='sos')
        # Filter Y
        Y_for_corr_map = signal.sosfilt(sos, Y)        
    return Y_for_corr_map

## Takes 2/3rds the time as using stats.pearsonr: 
def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)
def PLS_Pass(X,Y, tmp_x_for_plots, num_years = 50, Weight_Lat = True):        
    Y_for_corr_map = detrend_Y(Y, tmp_x_for_plots, Detrend_or_Filter = Detrend_or_Filter)
    # Get Filtered X for Correlation Matrix
    X_for_corr_map = detrend_X(X, tmp_x_for_plots, Detrend_or_Filter = Detrend_or_Filter)
    #Calculate correlation matrices of SLP and SNOTEL
    XY_corr = np.ones((X.shape[1]))
    for i in range(X.shape[1]):
        XY_corr[i] = np_pearson_cor(X_for_corr_map[:,i],Y_for_corr_map)
    if Weight_Lat == True:
        # # Christian 2016/Smoliak 2015
        # Before Projecting X onto Correlation Matrix (W), Weight W (or X) by Cosine of Lat and Standardize 
        ## populate weight array
        XY_corr_weighted_all_vars = np.ones(X_Flattened_Length)
        for grid_cells in range(len(X_var_list)):
            XY_corr_weighted = np.ones((X_dim_1,X_dim_2))
            XY_corr_reshape = XY_corr[(X_dim_1*X_dim_2)*grid_cells:(X_dim_1*X_dim_2)*(grid_cells+1)].reshape(X_dim_1,X_dim_2)
            for long in range(XY_corr_reshape.shape[1]):
                XY_corr_weighted[:,long] = XY_corr_reshape[:,long] * weights
            XY_corr_weighted = XY_corr_weighted.reshape(X_dim_1*X_dim_2)
            # Standardize
            XY_corr_weighted /= XY_corr_weighted.std()            
            # Add weighted correlation for variable 
            XY_corr_weighted_all_vars[(X_dim_1*X_dim_2)*grid_cells:(X_dim_1*X_dim_2)*(grid_cells+1)] = XY_corr_weighted        
    else:
        XY_corr_weighted_all_vars = XY_corr        
    XY_corr_weighted = XY_corr_weighted_all_vars    
    return XY_corr_weighted
def statspearsonr(X,Y, tmp_x_for_plots):        
    Y_for_corr_map = detrend_Y(Y, tmp_x_for_plots, Detrend_or_Filter = Detrend_or_Filter)
    # Get Filtered X for Correlation Matrix
    X_for_corr_map = detrend_X(X, tmp_x_for_plots, Detrend_or_Filter = Detrend_or_Filter)
    #Calculate correlation matrices of SLP and SNOTEL
    XY_corr = np.ones((X.shape[1]))
    XY_p = np.ones((X.shape[1]))
    for i in range(X.shape[1]):
        XY_corr[i],XY_p[i] = stats.pearsonr(Y_for_corr_map,X_for_corr_map[:,i])
    return XY_corr,XY_p
### Lots of Code for Simply Loading Sea Level Pressure (SLP) Data
### Code was previously set-up to also load/use geopotential height as a predictor,
### but has since been modified and is only up-to-date to use SLP
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
def smOLSconfidencelev(x,y,clev):
    res=sm.OLS(y, sm.add_constant(x)).fit()
    parmNconf=np.zeros([2])
    parmNconf[0]=res.params[1]
    parmNconf[1]=res.params[1]-res.conf_int(clev)[1][0]
    return parmNconf
def trenddetector(list_of_index, array_of_data, order=1):
    return float(np.polyfit(list_of_index, list(array_of_data), order)[-2])
def statslinregressts_slopeinterval(x,y,clev):
    mask = ~np.isnan(x) & ~np.isnan(y)
    ts= tinv(clev, sum(mask+0)-2)
    res = stats.linregress(x[mask], y[mask])
    return res.slope,ts*res.stderr
def define_coords(ds):
    if 'lat' not in ds.coords:
        ds.coords['lat'] = ds['lat']
    if 'lon' not in ds.coords:
        ds.coords['lon'] = ds['lon']
    return ds
def Load(X_var_list,stryr,endyr,m1,m2,latmin=20,Center=True,Scale=False):    
    if m1>m2:
        stryr1=stryr-1
    else:
        stryr1=stryr    
    ###SLP: https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html extended using ERA5
    prmsl = xr.open_dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/prmsl.mon.mean.nc');filstryr=1836;filendyr=2015 #20CRv3
    if endyr<filendyr+1:
        mslp=prmsl.prmsl[(stryr1-filstryr)*12:(endyr-filstryr+1)*12].sel(lat=slice(latmin,90))#20CRv3
    else:
        adjyr=10;adjstryr=filendyr-adjyr+1###adjust the ERA5 data by subtracting the 10-year mean monthly differences between the two reanalyses
        mslp=prmsl.prmsl[(stryr1-filstryr)*12:(filendyr-filstryr+1)*12].sel(lat=slice(latmin,90))#20CRv3
        mslp1=prmsl.prmsl[(adjstryr-filstryr)*12:(filendyr-filstryr+1)*12].sel(lat=slice(latmin,90)).mean(dim='time').to_numpy()
        prmsl1 = xr.open_mfdataset('/pscratch/sd/h/huoyilin/e5.sfc.msl.19402024.181x360.nc',preprocess=define_coords,);filstryr=1940            
        mslpera5=prmsl1.msl.swap_dims({"latitude": "lat"}).sel(lat=slice(latmin,90));prmsl1.close()
        mslp1-=mslpera5[(adjstryr-filstryr)*12:(filendyr-filstryr+1)*12].mean(dim='valid_time').to_numpy()
        mslp=np.concatenate((mslp, mslpera5[(filendyr+1-filstryr)*12:(endyr-filstryr+1)*12]+mslp1), axis=0)
        X_dim_1 = int(np.sum(prmsl.lat>=latmin))
    X_dim_2 = len(prmsl.lon)
    ## Get Weights for adjusting area by latitude
    weights = np.cos(np.deg2rad(prmsl.lat[prmsl.lat>=latmin]))
    weights = weights.values # array with dimension of latitude    
    # # ###ERA5
    # fn = '/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.t2m_msl_sp_siconc_tcw_tcwv.19402024.nc';filstryr=int(fn[-11:-7]);
    # ERA5_combine = xr.open_mfdataset(fn,combine='by_coords') ## open the data
    # # ERA5_combine =ERA5.sel(expver=1).combine_first(ERA5.sel(expver=5))
    # ERA5_combine.load()
    # mslp=ERA5_combine['msl'][(stryr1-filstryr)*12:(endyr-filstryr+1)*12].sel(latitude=slice(90,latmin))
    # lat=ERA5_combine['latitude']
    # X_dim_1 = int(np.sum(lat>=latmin))
    # X_dim_2 = len(ERA5_combine['longitude'])
    # del ERA5_combine
    # ### Create Array where X Data and Weights will be Stored
    # # All_X_Vars = np.zeros((num_years,X_dim_1*X_dim_2*len(X_var_list)))    
    # # Get Weights for adjusting area by latitude
    # weights = np.cos(np.deg2rad(lat[lat>=latmin]));del lat

    nyear = endyr-stryr+1
    X=seasonavg(mslp,m1,m2,nyear);del mslp
    ## Get Flattened Length to feed to other parts of code
    X_Flattened_Length = X_dim_1*X_dim_2*len(X_var_list)    
        # X = nov_march_mean_mslp.msl.values
    # PLS code centers already
    if Center == True: 
        X_mean = X.mean(axis=0)  ## gets mean for each grid cell across all years
        X -= X_mean
    ## Scale Data (code below is from source code of PLS)
    if Scale == True:
        X_std = X.std(axis=0, ddof=1)
        X_std[X_std == 0.0] = 1.0
        X /= X_std
        del X_std
    X = X.reshape(nyear,X_dim_1*X_dim_2)
    return X, X_Flattened_Length, X_dim_1, X_dim_2, weights
# ######## USER-DEFINED #######
Center = True               ## Whether or not to center X,Y Data
Scale =  True              ## Whether or not to scale X,Y Data
npass = 3              ## define how many passes want to make
Detrend_or_Filter = 'Linear_Detrend'   # 'Filter', 'Linear_Detrend', Diff_Detrend', 'FilterY_DetrendX', 'None'
X_var_list = ['SLP']        ## code was previously able to incorporate Geopotential Height, but currently set-up to work with SLP as predictor only
latmin = 20     ## Projecting X onto the weighted CC by the cosine of latitude over north of 20°N to form a time series and then be standardized
ols_or_sen = 'sen'          ## Use Sen's Slope or OLS when determining Slope of Qx1d Trends
######## USER-DEFINED #######
stryr=1959;endyr=2024

nyear=endyr-stryr+1
indexyr = np.arange(stryr,endyr+1)
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
    filstryr=1940;fn = '/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.t2m_msl_sp_siconc_tcw_tcwv.'+str(filstryr)+'2024.nc';sicnm='siconc'
    ERA5_combine = xr.open_mfdataset(fn,combine='by_coords') ## open the data    
    sic=ERA5_combine[sicnm][(adjstryr-filstryr)*12:(endyr-filstryr+1)*12].values;lat=ERA5_combine['latitude'].values;del ERA5_combine
    sic_areaavgera5=areaavg_lat(seasonavg(sic,m1,m2,endyr-adjstryr+1),lat);del sic
    sic_areaavg1-=np.mean(sic_areaavgera5[:adjyr],axis=0)
    sic_areaavg=np.concatenate((sic_areaavg,sic_areaavgera5[adjyr:]+sic_areaavg1), axis=0)
 
ds=nc.Dataset('/global/cfs/cdirs/m1199/huoyilin/cscratch/HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc')
lat=ds['latitude'][:];lon=ds['longitude'][:]
nlat=len(lat);nlon=len(lon)
t=ds['tas_mean'][(stryr1-1850)*12:(endyr-1849)*12]
t_areaavg=areaavg_lat(seasonavg(t,m1,m2,nyear),lat);del t
fn='/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.avg_slhtf_snlwrf_snlwrfcs_snswrf_snswrfcs_ishf_tnlwrf_tnlwrfcs_tnswrf_tnswrfcs.19502024.nc'
ds1=nc.Dataset(fn);year1=int(fn[-11:-7]);lat=ds1['latitude'][:]
# aht=(-ds1['mtnswrf'][:]+ds1['msnswrf'][:]-ds1['mtnlwrf'][:]+ds1['msnlwrf'][:]+ds1['mslhf'][:]+ds1['msshf'][:])[12*(stryr1-year1):12*(endyr-year1+1)]#-333550*(ds2['mtpr'][12*(stryr-filestryr):12*(endyr-filestryr+1)]))
aht=(-ds1['avg_tnswrf'][:]+ds1['avg_snswrf'][:]-ds1['avg_tnlwrf'][:]+ds1['avg_snlwrf'][:]+ds1['avg_slhtf'][:]+ds1['avg_ishf'][:])[12*(stryr1-year1):12*(endyr-year1+1)]#-333550*(ds2['mtpr'][12*(stryr-filestryr):12*(endyr-filestryr+1)]))
aht_areaavg=areaavg_lat(seasonavg(aht,m1,m2,nyear),lat);del aht
# ### Get Y ###
# ## Get Y Data into array format for PLS
ipredictand=3 ##0=SAT,1=SIC,2=ARs,3=AHT
if ipredictand==0:
    Y = t_areaavg+0
elif ipredictand==1:
    Y = sic_areaavg+0
elif ipredictand==2:
    Y = ar_areaavg+0
elif ipredictand==3:
    Y = aht_areaavg+0
if Center:
    Y-=np.mean(Y)
if Scale:
    Y/=np.std(Y, ddof=1)
### Get Y ###
# Get X
X, X_Flattened_Length, X_dim_1, X_dim_2, weights = Load(X_var_list,stryr,endyr,m1,m2,latmin,Center=Center,Scale=Scale,iensemble=iensemble,scenario=scenario)

#### INITIALIZE ARRAYS WHERE RESULTS ARE STORED ###
# +1 for overall variance
Y_dynadj = np.zeros((npass,nyear))
perc_var_expalained = np.zeros((npass+1))
XY_corr_weighted = np.zeros((npass,X_Flattened_Length))
t_scores = np.zeros((npass,nyear))
SLP_contribution = Y*0
### INITIALIZE ARRAYS WHERE RESULTS ARE STORED ###
# LOOP THROUGH EACH STATION AND APPLY PLS
current_pass = 0
####### FIRST PASS #########
print(f'Pass {current_pass}')
XY_corr_weighted[current_pass] = PLS_Pass(X,Y,indexyr)
for l in range(nyear):
    t_scores[current_pass,l]=np.mean(XY_corr_weighted[current_pass]*X[l,:])
t_scores[current_pass]/=np.std(t_scores[current_pass], ddof=1)    
#Calculate linear regression of t versus data being predicted
reg1=stats.linregress(t_scores[current_pass],Y)  
#Calculate SLP estimates of SNOTEL data
SLP_contribution = np.multiply(reg1[0],t_scores[current_pass])
#Subtract SLP estimates from Y data to yield dynamically-adjusted Y data
Y_dynadj[current_pass] = np.subtract(Y,SLP_contribution)
#square_error_temp[iyear] = np.square((Y_predict - Y_left_out))
# Save Percent Variance Explained
perc_var_expalained[current_pass] = np.var(SLP_contribution) / np.var(Y)

## NEEDED FOR SECOND PASS
current_pass += 1
if current_pass < npass:   
    # P = vector of regression coefficients for predictor variables X
    P_regression = np.ones((X_Flattened_Length))
    for i in range(X_Flattened_Length):
        P_regression[i] = stats.linregress(t_scores[current_pass-1],X[:,i])[0]
    # subtract X from SLP time-series multiplied by regression coefficients of X
    X_adj = X - np.dot(t_scores[current_pass-1].reshape(nyear,1),P_regression.reshape(X_Flattened_Length,1).T)        
#         # written out way to obtain tP rather than using matrix math
#         tP_long_form = np.ones((num_years,X.shape[1]))
#         for year in range(num_years):
#             for grid_cell in range(X.shape[1]):
#                 tP_long_form[year,grid_cell] = t_1[year] * P_regression[grid_cell]

    ####### SECOND PASS #########
    print(f'Pass {current_pass}')
    XY_corr_weighted[current_pass]= PLS_Pass(X_adj,Y_dynadj[current_pass-1],indexyr)
    for l in range(nyear):
        t_scores[current_pass,l]=np.mean(XY_corr_weighted[current_pass]*X_adj[l,:])
    t_scores[current_pass]/=np.std(t_scores[current_pass], ddof=1)    
    reg1=stats.linregress(t_scores[current_pass],Y_dynadj[current_pass-1,:])  
    #Calculate SLP estimates of SNOTEL data
    SLP_contribution_single_year = np.multiply(reg1[0],t_scores[current_pass])
    #Subtract SLP estimates from Y data to yield dynamically-adjusted Y data
    # get year that was predicted
    SLP_contribution+= SLP_contribution_single_year
    Y_dynadj[current_pass] = np.subtract(Y_dynadj[current_pass-1],SLP_contribution_single_year)
    perc_var_expalained[current_pass] = np.var(SLP_contribution_single_year) / np.var(Y)

    current_pass += 1
    if current_pass < npass:
        for i in range(X_Flattened_Length):
            P_regression[i] = stats.linregress(t_scores[current_pass-1],X_adj[:,i])[0]
        # subtract X from SLP time-series multiplied by regression coefficients of X
        X_adj_2 = X_adj-np.dot(t_scores[current_pass-1].reshape(nyear,1),P_regression.reshape(X_Flattened_Length,1).T)
        ####### THIRD PASS #########
        print(f'Pass {current_pass}')
        XY_corr_weighted[current_pass]= PLS_Pass(X_adj_2,Y_dynadj[current_pass-1],indexyr)
        for l in range(nyear):
            t_scores[current_pass,l]=np.mean(XY_corr_weighted[current_pass]*X_adj_2[l,:])
        t_scores[current_pass]/=np.std(t_scores[current_pass], ddof=1)    
        reg2=stats.linregress(t_scores[current_pass],Y_dynadj[current_pass-1,:])  
        #Calculate SLP estimates of SNOTEL data
        SLP_contribution_single_year = np.multiply(reg2[0],t_scores[current_pass])
        #Subtract SLP estimates from Y data to yield dynamically-adjusted Y data
        # get year that was predicted
        SLP_contribution += SLP_contribution_single_year
        Y_dynadj[current_pass] = np.subtract(Y_dynadj[current_pass-1],SLP_contribution_single_year)
        perc_var_expalained[current_pass] = np.var(SLP_contribution_single_year) / np.var(Y)


perc_var_expalained[npass] = np.var(SLP_contribution) / np.var(Y)
print('Done')
loadX1=True
np.savetxt('HadCRUT5NSIDC.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+season+'.dat', Y_dynadj[npass-1])
np.savetxt('HadCRUT5NSIDC.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+'Z'+season+'.dat', t_scores)
np.savetxt('HadCRUT5NSIDC.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+'perc_var_expalained'+season+'.dat', perc_var_expalained)
np.savetxt('HadCRUT5NSIDC.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+'SLP_contribution'+season+'.dat', SLP_contribution)
# np.savetxt('e5.sfc.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+season+'.dat', Y_dynadj[npass-1])
# np.savetxt('e5.sfc.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+'Z'+season+'.dat', t_scores)
# np.savetxt('e5.sfc.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+'perc_var_expalained'+season+'.dat', perc_var_expalained)
# np.savetxt('e5.sfc.lat70.'+str(stryr)+str(endyr)+season+'_ipredictand'+str(ipredictand)+'SLP_contribution'+season+'.dat', SLP_contribution)

fntsz=18;linew=3
if loadX1:
    X1, X_Flattened_Length, X_dim_1, X_dim_2, weights = Load(X_var_list,stryr,endyr,m1,m2,latmin,Center=True,Scale=False)
    X1=detrend_X(X1, indexyr, Detrend_or_Filter = Detrend_or_Filter)
    fn = '/global/cfs/cdirs/m1199/huoyilin/cscratch/e5.sfc.t2m_msl_sp_siconc_tcw_tcwv.19402024.nc';filstryr=int(fn[-11:-7])
    ds=nc.Dataset(fn);lat=ds['latitude'][:];lon=ds['longitude'][:];ds.close();del ds
    lat=lat[lat>=latmin]
    XY_corr,XY_p=statspearsonr(X1, Y,indexyr)
    loadX1=False
uneven_levels=np.concatenate((np.arange(-3,-.5), [-.5,-.2,.2,.5], np.arange(1,4)))
plt.figure(figsize=(16,16))
ax1=plt.subplot(2, 2, 1) # divide as 2x2, plot top left
ax1.set_title('(a) Regression coefficient',loc='left', fontsize=fntsz)
m = Basemap(projection='nplaea',boundinglat=25,lon_0=0,resolution='l',ax=ax1,round=True)
m.drawcoastlines(linewidth=.7, zorder=3)
xx, yy = m(np.meshgrid(lon,lat)[0],np.meshgrid(lon,lat)[1])
im=m.contourf(xx, yy, XY_corr.reshape(X_dim_1,X_dim_2),uneven_levels,cmap='seismic',extend='both')#,vmin=-5, vmax=0)
dd=10
for ilat in range(0,X_dim_1,dd):
    dd1=int(dd/np.cos(math.radians(lat[ilat])))
    m.plot(xx[ilat,::dd1],yy[ilat,::dd1],'.',color='black',markersize=4, zorder=1)
z_masked = np.ma.masked_where(XY_p < 0.01, XY_corr)
im=m.contourf(xx, yy, z_masked.reshape(X_dim_1,X_dim_2),uneven_levels,cmap='seismic',extend='both',zorder=2)#,vmin=-5, vmax=0)
cbar=m.colorbar(im);cbar.ax.tick_params(labelsize=fntsz);cbar.set_label(label='hPa',size=fntsz)
Y_detrended=detrend_Y(Y, indexyr, Detrend_or_Filter = Detrend_or_Filter)
if Scale:
    Y_detrended/=np.std(Y_detrended, ddof=1)
X_composite=np.zeros((2,X_Flattened_Length))
count=np.zeros((2))
for iyear in range (0,nyear):
    if Y_detrended[iyear] > 1:
        X_composite[0]+=X1[iyear]
        count[0]+=1
    elif Y_detrended[iyear] < -1:
        X_composite[1]+=X1[iyear]
        count[1]+=1
X_composite[0]/=count[0]
X_composite[1]/=count[1]
uneven_levels=np.concatenate((np.arange(-6,-1,2), [-1,-.4,.4,1], np.arange(2,8,2)))
ax1=plt.subplot(2, 2, 2) # divide as 2x2, plot top right
ax1.set_title('(b) Composite difference',loc='left', fontsize=fntsz)
m = Basemap(projection='nplaea',boundinglat=25,lon_0=0,resolution='l',ax=ax1,round=True)
m.drawcoastlines(linewidth=.7, zorder=3)
xx, yy = m(np.meshgrid(lon,lat)[0],np.meshgrid(lon,lat)[1])
im=m.contourf(xx, yy, (X_composite[0]-X_composite[1]).reshape(X_dim_1,X_dim_2)/100,uneven_levels,cmap='seismic',extend='both')#,vmin=-5, vmax=0)
cbar=m.colorbar(im);cbar.ax.tick_params(labelsize=fntsz);cbar.set_label(label='hPa',size=fntsz)
weights = np.cos(np.deg2rad(lat));weights = np.expand_dims(weights, axis=1)
weights = np.repeat(weights, X_dim_2, axis=1).reshape(X_dim_1*X_dim_2)
weights=np.where(weights<0,0,weights)
print(str(np.cov(XY_corr, X_composite[0]-X_composite[1], aweights=weights)[0][1] / np.sqrt(np.cov(XY_corr, XY_corr, aweights=weights)[0][1] * np.cov(X_composite[0]-X_composite[1], X_composite[0]-X_composite[1], aweights=weights)[0][1])))
ax1=plt.subplot(2, 1, 2) # divide as 2x1, plot bottom
ax1.set_title('(c)',loc='left', fontsize=fntsz)
if ipredictand==0:
    tmp_areaavg1 = t_areaavg+0;predictandname='SAT';unit='K'
elif ipredictand==1:
    tmp_areaavg1 = sic_areaavg+0;predictandname='SIC';unit='%'
elif ipredictand==2:
    tmp_areaavg1 = ar_areaavg*100;predictandname='AR';unit='%'
elif ipredictand==3:
    predictandname='PET'
    tmp_areaavg1 = aht_areaavg+0;unit='W m$^{-2}$'
if Center: tmp_areaavg1-=np.mean(tmp_areaavg1)
ax1.plot(indexyr, tmp_areaavg1,'k',lw=linew, label='Raw')#*t_trend_contribution[0]
t_scores_T=t_scores.T
reg = LinearRegression().fit(t_scores_T, tmp_areaavg1)
predict=reg.predict(t_scores_T)
ax1.plot(indexyr, predict,'b',lw=linew, label='Dynamically-induced ')#*t_trend_contribution[0]
ax1.plot(indexyr,tmp_areaavg1-predict,'r',lw=linew, label='Residual')#*t_trend_contribution[0]
ax1.set_xlim(stryr, endyr)
ax1.set_ylabel(predictandname+' anomalies ('+unit+')', fontsize=fntsz)
ax1.legend(frameon=False,labelcolor='linecolor', fontsize=fntsz,handlelength=0,ncols=3)
ax1.tick_params(axis='both', which='major',length=10, labelsize=fntsz)
ax1.xaxis.set_minor_locator(AutoMinorLocator());ax1.yaxis.set_minor_locator(AutoMinorLocator())        
ax1.tick_params(axis='both', which='minor',length=5)
pv=stats.linregress(tmp_areaavg1, predict).pvalue
if pv>=0.0005:ax1.text(indexyr[0]+1.5, np.max(ax1.get_ylim())/4*3, "R$^{2}$ (black, blue) = %.2f (p = %.3f)" %(np.corrcoef(tmp_areaavg1,predict)[0,1],pv),fontsize=fntsz)
else:ax1.text(indexyr[0]+1.5, np.max(ax1.get_ylim())/4*3, "R$^{2}$ (black, blue) = %.2f " %np.corrcoef(tmp_areaavg1,predict)[0,1],fontsize=fntsz)
ax1.axhline(y = 0, color = 'k', alpha = .5,lw=linew/3)
plt.show()
fn='/pscratch/sd/h/huoyilin/e3sm_scratch/pm-cpu/Fig3.nc'
if os.path.exists(fn):        os.remove(fn)
ncfile = nc.Dataset(fn,mode='w',format='NETCDF4_CLASSIC') 
ncfile.comment='Data for Huo et al. (2025)'
latitude_dim = ncfile.createDimension('latitude',len(lat)) # unlimited axis (can be appended to).
longitude_dim = ncfile.createDimension('longitude',len(lon)) # unlimited axis (can be appended to).
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
time = ncfile.createVariable('time', np.float64, ('time',));time.units = 'years since 1979';time[:]=indexyr-stryr+1
latitude = ncfile.createVariable('latitude', np.float64, ('latitude',));latitude.units = 'degrees_north';latitude[:]=lat
longitude = ncfile.createVariable('longitude', np.float64, ('longitude',));longitude.units = 'degrees_east';longitude[:]=lon
# Define a 2D variable to hold the data
coef_SLP_PET = ncfile.createVariable('coef_SLP_PET',np.float64,('latitude','longitude')) # note: unlimited dimension is leftmost
coef_SLP_PET.long_name = 'Regression coefficient of detrended DJF-mean sea level pressure anomalies upon the standardized detrended PET index' # this is a CF standard name
coef_SLP_PET[:]=XY_corr.reshape(X_dim_1,X_dim_2)
p_values_SLP_PET = ncfile.createVariable('p_values_SLP_PET',np.float64,('latitude','longitude')) # note: unlimited dimension is leftmost
p_values_SLP_PET.long_name = 'Significance levels of regression coefficient of detrended DJF-mean sea level pressure anomalies upon the standardized detrended PET index' # this is a CF standard name
p_values_SLP_PET[:]=XY_p.reshape(X_dim_1,X_dim_2)
SLP_composite_diff = ncfile.createVariable('SLP_composite_diff',np.float64,('latitude','longitude')) # note: unlimited dimension is leftmost
SLP_composite_diff.long_name = 'Composite difference of detrended DJF-mean SLP anomalies averaged over years wherein the standardized detrended PET index exceeds one standard deviation.' # this is a CF standard name
SLP_composite_diff[:]=(X_composite[0]-X_composite[1]).reshape(X_dim_1,X_dim_2)/100;SLP_composite_diff.units='hPa'
PET_raw = ncfile.createVariable('PET_raw',np.float64,('time')) # note: unlimited dimension is leftmost
PET_raw.long_name = 'Raw DJF-mean PET anomaly time series over the Arctic' # this is a CF standard name
PET_raw[:]=tmp_areaavg1;PET_raw.units='W m-2'
PET_dynamically_induced  = ncfile.createVariable('PET_dynamically_induced',np.float64,('time')) # note: unlimited dimension is leftmost
PET_dynamically_induced.long_name = 'Dynamically induced DJF-mean PET anomaly time series over the Arctic' # this is a CF standard name
PET_dynamically_induced[:]=predict;PET_dynamically_induced.units='W m-2'
ncfile.close()
