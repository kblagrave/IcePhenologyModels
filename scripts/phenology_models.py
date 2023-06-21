
import pandas as pd
import numpy as np

import pickle

import datetime

from pathlib import Path

import argparse


# relative location of RandomForest models
MODEL_DIR = Path('../Models')


suffix = '_v11' # this is the most recent model result


# find seasonal limits based on training model data
X = pd.read_csv(MODEL_DIR/f'rf_iceoff_final{suffix}_input.csv')
maxSON, maxDJF, maxMAM, maxJJA = X['TMINMAX_lagSON'].max(), X['TMINMAX_DJF'].max(), X['TMINMAX_MAM'].max(), X['TMINMAX_lagJJA'].max()
minSON, minDJF, minMAM, minJJA = X['TMINMAX_lagSON'].min(), X['TMINMAX_DJF'].min(), X['TMINMAX_MAM'].min(), X['TMINMAX_lagJJA'].min()


def predict_phenology(lat, lon, year, verbose = True, depth_avg = None, 
                       extrapolation = False,
                      lake_area = None, shore_dev = None,  elevation=None,
                     filename=None, 
                     seasonalT=None,
                     debug= False):
    """
    lat: latitude of lake
    lon: longitude of lake
    year: end of winter season (i.e., 1990 for winter of 1989-1990); 

    verbose: print progress to screen

    filename: a file with monthly temperatures in columns Tave01 to Tave12 for each Year, Latitude, Longitude
        required columns: Latitude, Longitude, Year, Tave01, Tave02, ... Tave12

    seasonalT: list of [summer,fall,winter,spring] mean seasonal temperatures for given lat,lon and year.
    
    OUTPUT: ice on and ice off dates, as day of year.

    """

    #climatena_filename = CLIMATENA_DIR/f'US_icephenologymodels74lakes_{model.upper()}_{ssp}_2011-2100MP.csv'
    
    # read in models
    with open(MODEL_DIR/f'rf_iceon_final{suffix}.pickle', 'rb') as f:
        rf_iceon = pickle.load(f)
    with open(MODEL_DIR/f'rf_iceoff_final{suffix}.pickle', 'rb') as f:
        rf_iceoff = pickle.load(f)
    
    if isinstance(year, list):
        maxyear = np.max(year)
    else:
        maxyear = year
        
    latlon = f'{lat:+07.3f}{lon:+08.3f}'
    

    """ This code can be modified if there is a locally installed HydroLAKES shapefile """
    """ HydroLAKES shapefiles contain all the necessary lake morphology data """
    if (depth_avg is None) | (lake_area is None) | (shore_dev is None) | (elevation is None):
        try:
            # Find matching HydroLAKES shapefile
            hylak_dist, hylak_id = match2hydrolakes(lat, lon)
        except:
            print('You need to supply depth_avg, lake_area, shore_dev, and elevation.')
            #print(f'\tClosest HydroLAKES lake {hylak_id} is {hylak_dist:.1f} km away.')
            return np.nan, np.nan
        if np.isnan(hylak_dist):
            print('No HydroLAKES match at this coordinate. You need to supply depth_avg, lake_area, shore_dev, and elevation.')
            #print(f'\tClosest HydroLAKES lake {hylak_id} is {hylak_dist:.1f} km away.')
            return np.nan, np.nan
        lakechar = get_hylak_info(hylak_id)
    else:
        lakechar = dict(Depth_avg=depth_avg, Lake_area=lake_area, Shore_dev=shore_dev, Elevation=elevation)
    
    elevation = lakechar['Elevation']
    
    # earliest and latest day of year used in model prediction
    # 1 June, May 31
    list_of_leap_years = [i for i in range(1800,2021,4) if (
                                                ((i % 400)==0) | 
                                                    ((
                                                        (i % 100)!=0) & ((i % 4)==0)))]
    earliest = -213
    if (maxyear+1) in list_of_leap_years:
        latest = 152
    else:
        latest = 151
    
    if isinstance(year, list):
        start_date = (pd.to_datetime(f'31 December {np.min(year)}') + pd.to_timedelta(f'{earliest} days')).strftime('%d %B, %Y')
        end_date = (pd.to_datetime(f'31 December {np.max(year)}') + pd.to_timedelta(f'{latest} days')).strftime('%d %B, %Y')    
    else:
        start_date = (pd.to_datetime(f'31 December {year}') + pd.to_timedelta(f'{earliest} days')).strftime('%d %B, %Y')
        end_date = (pd.to_datetime(f'31 December {year}') + pd.to_timedelta(f'{latest} days')).strftime('%d %B, %Y')
    
    if verbose:
        print(f'Building weather time series for coordinate ({lat:.3f},{lon:.3f})')
        print(f'  {start_date} to {end_date}') 
    
    # read in data from ClimateNA files (using model and ssp time series)
    # tavg from monthly to seasonal columns (Tave01, Tave02, etc.)
    if (filename is None) & (seasonalT is None):
        print('Need to specify filename. File must have monthly temperatures in columns Tave01, Tave02, etc.')
        return np.nan, np.nan
    elif seasonalT is not None:
        tavg = pd.DataFrame([seasonalT], index = [latlon], columns=['TMINMAX_lagJJA','TMINMAX_lagSON','TMINMAX_DJF','TMINMAX_MAM'])
    else:
        tavg = pd.read_csv(filename)            
    
    # modify this to find closest match
        if debug:
            print(lat,lon, year)
        ind = (tavg.Latitude == lat) & (tavg.Longitude == lon) & (tavg.Year.isin([year,year-1]))
        if debug:
            print('equal',ind.sum())
        ind = np.isclose(tavg.Latitude,lat) & np.isclose(tavg.Longitude,lon) & (tavg.Year.isin([year,year-1]))
        if debug:
            print('close',ind.sum())
        tavg = tavg.loc[ind,:]
        for mm in range(6,13):
            tavg[f'Tave_lag{mm:02d}'] = tavg[f'Tave{mm:02d}'].shift()
        #display(tavg[['Year','Latitude','Longitude','ID1','ID2']+[c for c in tavg.columns if ('Tave' in c)]])
        tavg = tavg.loc[tavg.Year==year,:]
        
        tavg['TMINMAX_lagJJA'] = (tavg[f'Tave_lag06'] + tavg[f'Tave_lag07'] + tavg[f'Tave_lag08'])/3.
        tavg['TMINMAX_lagSON'] = (tavg[f'Tave_lag09'] + tavg[f'Tave_lag10'] + tavg[f'Tave_lag11'])/3.
        tavg['TMINMAX_DJF'] = (tavg[f'Tave_lag12'] + tavg[f'Tave01'] + tavg[f'Tave02'])/3.
        tavg['TMINMAX_MAM'] = (tavg[f'Tave03'] + tavg[f'Tave04'] + tavg[f'Tave05'])/3.
        tavg = tavg[['TMINMAX_lagJJA','TMINMAX_lagSON','TMINMAX_DJF', 'TMINMAX_MAM']]
        tavg.index = [latlon]
        
    if verbose:
        print(tavg)
    
    if len(tavg.dropna(how='all',axis=0))==0:
        return np.nan, np.nan    

    #features = ['Shore_len', 'Depth_avg']
    features = ['Depth_avg', 'Lake_area']
    features_on = [i for i in rf_iceon.feature_names_in_ if i not in tavg.columns]
    features_off = [i for i in rf_iceoff.feature_names_in_ if i not in tavg.columns]
    
    iceon_multiple, iceoff_multiple = [],[]
    # run through all rows of tavg
    for i,row in tavg.iterrows():
        
        X = tavg.loc[[i],:].copy()
        X.index = [latlon]
        
        #print(features_on)
        #print(features_off)
    
        Xon = pd.concat([X,pd.DataFrame(lakechar,index=[latlon])[features_on]],ignore_index=False,axis=1)
        Xon.columns = Xon.columns.astype(str)
        Xon = Xon[rf_iceon.feature_names_in_]

        Xoff = pd.concat([X,pd.DataFrame(lakechar,index=[latlon])[features_off]],ignore_index=False,axis=1)
        Xoff.columns = Xoff.columns.astype(str)
        Xoff = Xoff[rf_iceoff.feature_names_in_]
    
    #if rf_model == 'daily':
    #    iceon = int(np.round(rf_iceon.predict(Xon)))
    #else:
    #    iceon = int(np.round(rf_iceon.predict(Xon.drop('TMINMAX_MAM',axis=1))))

        iceon = int(np.round(rf_iceon.predict(Xon)))

        iceoff = int(np.round(rf_iceoff.predict(Xoff)))
    
        if not extrapolation:
            if (tavg['TMINMAX_lagJJA'].between(minJJA, maxJJA).all() & 
                tavg['TMINMAX_lagSON'].between(minSON, maxSON).all() &
                tavg['TMINMAX_DJF'].between(minDJF, maxDJF).all() &
                tavg['TMINMAX_MAM'].between(minMAM, maxMAM).all()):
                iceoff = iceoff
            elif (tavg['TMINMAX_lagJJA'].between(minJJA, maxJJA).all() & 
                tavg['TMINMAX_lagSON'].between(minSON, maxSON).all() &
                tavg['TMINMAX_DJF'].between(minDJF, maxDJF).all()):
                iceon = iceon
                iceoff = np.nan
            else:
                iceon = np.nan
                iceoff = np.nan
        
        iceon_multiple.append(iceon)
        iceoff_multiple.append(iceoff)
    
    if isinstance(year,list):
        return iceon_multiple, iceoff_multiple
    else:
        return iceon_multiple[0], iceoff_multiple[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # optional summer, fall, winter, spring temperature OR filename
    selectNamed = parser.add_argument_group('select one of two arguments')
    selectNamed.add_argument("-f", "--filename",dest="filename",required=False,help="Monthly temperature csv file.")
    selectNamed.add_argument("-t", "--temperatures",dest="seasonalT",nargs=4,type=float,required=False,help="Seasonal temperatures (summer, fall, winter, spring).")

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-y", "--year",dest ="year", required=True,help="Single year (e.g., 1990 for winter season 1989-1990)")
    requiredNamed.add_argument("-c", "--coordinates", nargs=2,type=float, dest="coo",required=True,help="Enter latitude and longitude of lake")
    requiredNamed.add_argument("-d", "--depth_avg", dest='depth',required=True,help="Average lake depth")
    requiredNamed.add_argument("-e", "--elevation", dest='elevation',required=True,help="Elevation")
    requiredNamed.add_argument("-s", "--shore_dev", dest='shore_dev',required=True,help="Shoreline development (1=circle)")
    requiredNamed.add_argument("-a", "--lake_area", dest='area',required=True,help="Surface area")

    args = parser.parse_args()

    lat,lon = args.coo
    year = int(args.year) #[int(y) for y in args.year.split('-')]
    filename = args.filename
    seasonalT = args.seasonalT
    depth,area,shore_dev,elevation = args.depth,args.area,args.shore_dev,args.elevation

    iceon,iceoff = predict_phenology(lat,lon, year, filename=filename,
                                     seasonalT = seasonalT,
                                     verbose=False,
                                      depth_avg = depth,
                                      lake_area = area, 
                                      shore_dev = shore_dev,
                                      elevation = elevation,)
    
    
    print('\n'+'='*40)
    print(f'Lake coordinate: {lat}, {lon}')
    print(f'Lake depth: {depth} m')
    print(f'Lake surface area: {area} sq km')
    print(f'Shoreline development: {shore_dev}')
    print(f'Elevation: {elevation} m\n')

    print(f'Winter of {year-1}-{year}')
    print('-'*19)

    if not np.isnan(iceon):
        iceondate = (pd.to_datetime(f'Dec 31, {year-1}') + pd.to_timedelta(f'{iceon} days')).strftime('%B %d, %Y')
    else:
        iceondate = 'Unknown'
    if not np.isnan(iceoff):
        iceoffdate = (pd.to_datetime(f'Dec 31, {year-1}') + pd.to_timedelta(f'{iceoff} days')).strftime('%B %d, %Y')
    else:
        iceoffdate = 'Unknown'

    print(f'Predicted ice-on date (day of year) : {iceondate} ({iceon})')
    print(f'Predicted ice-off date (day of year): {iceoffdate} ({iceoff})\n')