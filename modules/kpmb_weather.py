import pandas as pd
import numpy as np
from geopy import distance
import os
from pathlib import Path

import requests

volume = 'My Passport for Mac'

OUTPUTDIR = Path(f'/Volumes/{volume}/WeatherData/NOAA/csv/')


def download(url, outfile, chunk_size=128, header=None, verify=True):
    r = requests.get(url, stream=True,verify=verify)
    with open(outfile, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)


def get_date(datestring,df,after=None, before=None, variable = 'ave_air_temp_adjusted', label=None, datecolumn=None):
    """
    =====
    Extract data from dataframe, df for dates before or after a given date (datestring)
    =====
    datestring: date string in format interpretable by pd.to_datetime
    df: dataframe
    after: days after
    before: days before
    variable: name of column from which data is being extracted
    datecolumn: column of df which contains date information
    label: index of returned dataframe row
    Output: returns a single row dataframe with after+before+1 columns for each of the days for given variable
    """

    if before is None:
        before =0
    if after is None:
        after = 0
    if label is None:
        label = 0
    columns = ['{}_{:+04d}day'.format(variable,c) for c in range(-before,after+1)]
    if datecolumn is None:
        datecolumn = [c for c in df.columns if 'date' in str(c).lower()]
    if isinstance(datecolumn,list):
        datecolumn = datecolumn[0]

    # need to sort by datecolumn!
    dfworking = df.sort_values(datecolumn).reset_index(drop=True)
    first_date = pd.to_datetime(min(dfworking[datecolumn]))
    last_date = pd.to_datetime(max(dfworking[datecolumn]))
    if not isinstance(datestring,str):
        result = pd.DataFrame([[np.nan]*(before+after+1)], columns=columns, index=[label])
    else:
        dfresult = dfworking.copy()
        dfresult[datecolumn] = pd.to_datetime(dfworking[datecolumn])
        dfresult = dfresult.sort_values(datecolumn)
        dfresult = dfresult.set_index(datecolumn)
        if pd.to_datetime(datestring) in dfresult.index:
            start_date0 = pd.to_datetime(datestring)-pd.to_timedelta('{} days'.format(before))
            end_date0 = pd.to_datetime(datestring) + pd.to_timedelta('{} days'.format(after))

            start_date = max([first_date,start_date0])
            end_date = min([last_date,end_date0])
            result = dfresult.loc[start_date:end_date,variable]
            result = result.reindex(pd.date_range(start_date0,end_date0))
#            if start_date != dfresult.index[0]:
#                # pad result with NaN
#                result = result[pd.date_range(start_date,result.index[-1])]
#            if end_date != dfresult.index[-1]:
#                # pad result with NaN
#                result = result[pd.date_range()]

            result = pd.DataFrame([result.values], columns=columns, index=[label])
        else:
            result = pd.DataFrame([[np.nan]*(before+after+1)], columns=columns, index=[label])
    return result

# find closest stations based on dfstations.LAT_DEC, dfstations.LON_DEC
def find_stations(lat,lon,dfstations,dist=50,elev=0,d_elev=1e6,
                    start_date='1 Jan 1800', end_date='31 Dec 2021',
                    begin_column='BEGIN_DATE',end_column='END_DATE',verbose=True,
                    elevcolumn='ELEV_GROUND',
                    latcolumn='LAT_DEC',loncolumn='LON_DEC', uniqueID='NCDCSTN_ID'):

    date_range = pd.date_range(start_date,end_date,freq='D')
    dlat = dist / 111.
    dlon = dlat/np.cos(lat*np.pi/180.)
    ind = (dfstations[latcolumn].between(lat-dlat,lat+dlat) &
           dfstations[loncolumn].between(lon-dlon,lon+dlon) &
           dfstations[elevcolumn].between(elev-d_elev,elev+d_elev)
           )
    #print(elev-d_elev, elev+d_elev)
    #print(ind.sum())
    dfsub = dfstations[ind].copy()
    if len(dfsub)==0:
        return np.nan, np.nan, np.nan
    #display(dfsub)
    #ind = (dfsub.BEGIN_DATE==10101)
    #for i,row in dfsub[ind].iterrows():
    #    source_id_file = f'{OUTPUTDIR}{row.SOURCE_ID}.csv'
    #    try:
    #        df_ = pd.read_csv(source_id_file)
    #        earliest_date = df_.DATE.min()
    #        latest_date = df_.DATE.max()
    #        dfsub.loc[i,'BEGIN_DATE'] = earliest_date.replace('-','')
    #        dfsub.loc[i,'END_DATE'] = latest_date.replace('-','')
    #    except:
    #        pass

    # find distance from this subset of stations
    dfsub['Distance_km'] = dfsub.apply(lambda row: distance.distance((row[latcolumn],row[loncolumn]),(lat,lon)).km,axis=1)
    dfsub = dfsub.sort_values('Distance_km')
    dfsub = dfsub[dfsub.Distance_km<=dist]

    #display(dfsub)

    dfsub = find_by_date(dfsub,start_date=start_date,
                        end_date=end_date,begin_column=begin_column,
                        end_column=end_column,uniqueID=uniqueID)
    #display(dfsub)
    # some BEGIN_DATE (10101) and END_DATE (99991231) may be wrong
    #display(dfsub)

    stations = {}
    distances = {}
    if verbose:
        print('Total number of days:',len(date_range))
    while True:
        #
        if (len(dfsub)==0) | (dfsub.NFRACTION.max()==0):
            break
        select_station = dfsub.sort_values('NFRACTION',ascending=False)[uniqueID].values[0]
        dfworking = dfstations[dfstations[uniqueID] == select_station]

        try:
            other_stn_name = ', '.join(dfstations.loc[dfstations[uniqueID]==select_station,'GHCND_ID'].drop_duplicates().tolist())
            # print(other_stn_name)
        except:
            other_stn_name = 'NONE'
        distance_away = dfsub.loc[dfsub[uniqueID]==select_station,'Distance_km'].values[0]
        if verbose:
            print('Station: {} ({:.1f} km from coordinate) aka {}'.format(select_station,
                                            distance_away, other_stn_name))
            print('\tCoverage of (remaining) years: {:.2f}%'.format(dfsub.NFRACTION.max()*100))
        datekeep = []
        for i, row in dfworking.iterrows():
            # remove select rows from date_range
            workingrange = pd.date_range(pd.to_datetime(row[begin_column],format='%Y%m%d'),
                                         pd.to_datetime(str(row[end_column]).replace('9999','2021'),format='%Y%m%d'),freq='D')
            overlap_range = list(set(workingrange) & set(date_range))
            overlap_range.sort()
            datekeep.append(overlap_range)
            #print(workingrange,date_range)
            if (len(overlap_range)>0) & verbose:
                print('\t\t{}-{}'.format(min(overlap_range).strftime('%Y/%m/%d'),max(overlap_range).strftime('%Y/%m/%d')))
            date_range = set(date_range) - set(workingrange)#list(set(workingrange) & set(date_range))
            #print('removing ',workingrange)
        stations[select_station] = datekeep
        distances[select_station] = distance_away
        #remove station from dfsub
        dfsub = dfsub.loc[dfsub[uniqueID]!=select_station,:]
        if verbose:
            print('\n')
            print('Number of missing days remaining:', len(date_range))
        if len(date_range)==0:
            break
        # USE date_range to adjust NFRACTION in dfsub
        for i,row in dfsub.iterrows():
            # find all rows with that NCDCSTN_ID
            dfoo = dfstations[dfstations[uniqueID]==row[uniqueID]]
            # run through all of these rows, finding the total number of overlap days
            ndays = 0
            useddays = []
            for ii,drow in dfoo.iterrows():
                try:
                    # begin date can be 10101
                    workingrange = pd.date_range(pd.to_datetime(drow[begin_column],format='%Y%m%d'),pd.to_datetime(str(drow[end_column]).replace('9999','2021'),format='%Y%m%d'),freq='D')
                except ValueError:
                    continue
                ndays += len(list((set(date_range) & set(workingrange)) - set(useddays)))
                useddays = set(workingrange) | set(useddays)
            dfsub.loc[i,'NFRACTION'] = ndays/len(date_range)
    if (len(date_range)!=0) & verbose:
        print('\n\nConsider increasing the distance to the nearest weather station.')
        print('   Missing data for {} days'.format(len(date_range)))
    return stations, len(date_range), distances

def get_id(lat,lon,dfstations,dist=50, elev=0,d_elev=1e6,getid='GHCND_ID',
                latcolumn='LAT_DEC',loncolumn='LON_DEC',elevcolumn='ELEV_GROUND'):
    #dlat = dist / 111.
    #dlon = dlat/np.cos(lat*np.pi/180.)
    #ind = dfstations.LAT_DEC.between(lat-dlat,lat+dlat) & dfstations.LON_DEC.between(lon-dlon,lon+dlon)
    #dfsub = dfstations[ind].copy()
    #dfsub['Distance_km'] = dfsub.apply(lambda row: distance.distance((row.LAT_DEC,row.LON_DEC),(lat,lon)).km,axis=1)
    dfsub = filter_stations_list(lat,lon,dfstations, latcolumn=latcolumn, loncolumn=loncolumn,elevcolumn=elevcolumn,elev=elev,d_elev=d_elev,dist=dist)
    #dfsub.sort_values('Distance_km')
    #print(elev-d_elev, elev+d_elev)
    if len(dfsub)==0:
        display(dfsub)
        result = dfsub[getid].copy()
    else:
        result = dfsub[getid].dropna().drop_duplicates() #unique()

    return result

def filter_stations_list(lat,lon,dfstations, latcolumn='LAT_DEC',loncolumn='LON_DEC', elevcolumn='ELEV_GROUND',
                dist=50,elev=0,d_elev=1e6):
    dlat = dist / 111.
    dlon = dlat/np.cos(lat*np.pi/180.)
    ind = (dfstations[latcolumn].between(lat-dlat,lat+dlat) &
            dfstations[loncolumn].between(lon-dlon,lon+dlon) &
            dfstations[elevcolumn].between(elev-d_elev,elev+d_elev)
            )
    dfsub = dfstations[ind].copy()
    #print(dfsub[elevcolumn].unique())
    if len(dfsub)==0:
        dfsub['Distance_km'] = np.nan
        return dfsub

    dfsub['Distance_km'] = dfsub.apply(lambda row: distance.distance((row[latcolumn],row[loncolumn]),(lat,lon)).km,axis=1)
    dfsub = dfsub.sort_values('Distance_km')
    #display(dfsub[elevcolumn].sort_values().unique())

    dfsub = dfsub[(dfsub.Distance_km<=dist)] # & (dfsub[elevcolumn].between(elev-d_elev,elev+d_elev))]

    #display(dfsub[elevcolumn].sort_values().unique())
    return dfsub

def download_weather_data(lat,lon, dfstations,dist=15,start_date='1 Jan 1800', download_all=False,
                          end_date = '31 Dec 2021',exact_coverage=False, verbose=True, includeAttributes='0'):

    baseurl = 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries'


    if download_all:
        ghcnd_id = get_id(lat,lon,dfstations)
        if verbose:
            print(f'{len(ghcnd_id)} stations within {dist} km of ({lat:.3f},{lon:.3f})')
        for ghcnd in ghcnd_id:
            # outfile = '/Volumes/Data/WeatherData/NOAA/csv/{}.csv'.format(ghcnd)

            outfile = OUTPUTDIR/f'{ghcnd}.csv'
            if os.path.exists(outfile):
                continue
            weatherurl = baseurl+'&stations={}&includeAttributes={}&startDate={}&'\
                'endDate={}&dataTypes={},{},{},{},{},{},{},{},{},{},{}&format={}&units=metric'.format(
                ghcnd,includeAttributes,'1700-01-01','2030-12-31','TOBS','TAVG','TMIN','TMAX',
                'AWDR','AWND','PRCP','SNOW','SNWD','THIC','TSUN','csv')
            download(weatherurl, outfile)
            if verbose:
                print(f'Saved to {outfile}')
        return

    # the following may be broken (10 FEB 2022 - KB)

    # Get list of stations to use for each date
    stations, missingdays = find_stations(lat,lon,dfstations,
                      dist=dist,
                      start_date=start_date, end_date = end_date)

    # some ice records go back longer than weather station data
    # so if days are missing only at the beginning of the record, it's not a problem
    earliest_date = pd.to_datetime('31 Dec 2030')
    for key,value in stations.items():
        early_v = min([i for v in value for i in v])
        earliest_date = min(earliest_date,early_v)
    missing_early_days = (earliest_date - pd.to_datetime(start_date)).days

    print(missing_early_days)

    if (missingdays!=0) & (missingdays != missing_early_days) & (exact_coverage):
        print('Increase distance to fill in missing record.')
        return
    #return stations
    # website to access data


    # run through all stations matched to lat,lon pair
    for station,dates in stations.items():
        # need GHCND_ID to request data
        ind = dfstations.NCDCSTN_ID == station
        ghcnd_id = dfstations.loc[ind,'GHCND_ID'].dropna().drop_duplicates().values
        # if there are two ghcnd_id or no ghcnd_id associated with NCDCSTN_ID then
        #    there is a problem??
        if len(ghcnd_id) != 1:
            print('There is a problem with the GHCND_ID')
            print('NCDCSTN_ID: {}'.format(station))
            print('GHCND_ID: {}'.format(','.join(ghcnd_id)))
        else:
            pass
            #ghcnd_id = ghcnd_id[0]

        # check if we can just make one call to API instead of multiple calls
        alldates = [d for dd in dates for d in dd]
        alldates.sort()
        missing_dates = pd.date_range(min(alldates),max(alldates)).difference(alldates)
        # if there is no difference between alldates and sum of 'dates' then just make
        #    one call to api by sending alldates. Otherwise cycle through
        if len(missing_dates)==0:
            # still needs to be a list of lists for for loop below
            alldates = [alldates]
        else:
            alldates = dates.copy()

        for date in alldates:
            if len(date)==0:
                continue
            start_date = min(date).strftime('%Y-%m-%d')
            end_date = max(date).strftime('%Y-%m-%d')
            for ghcnd in ghcnd_id:
                outfile = OUTPUTDIR/f'{ghcnd}_{start_date}_{end_date}.csv' 
                #outfile = '{}{}_{}_{}.csv'.format(OUTPUTDIR,ghcnd,start_date,end_date)


#                outfile = '/Volumes/Data/WeatherData/NOAA/csv/{}_{}_{}.csv'.format(ghcnd,start_date,end_date)
                if os.path.exists(outfile):
                    continue
                weatherurl = baseurl+'&stations={}&includeAttributes={}&startDate={}&'\
                    'endDate={}&dataTypes={},{},{},{},{},{},{},{},{},{},{}&format={}&units=metric'.format(
                    ghcnd,includeAttributes,start_date,end_date,'TOBS','TAVG','TMIN','TMAX',
                    'AWDR','AWND','PRCP','SNOW','SNWD','THIC','TSUN','csv')
                download(weatherurl, outfile)
                if verbose:
                    print(f'Saved to {outfile}')



def find_by_date(df, start_date='1 Jan 1800', end_date='31 Dec 2021',
                        uniqueID='NCDCSTN_ID', begin_column='BEGIN_DATE',
                        end_column='END_DATE'):

    # df is DataFrame list of stations from NOAA
    dfresult = df.copy()
    for name,group in df.groupby(uniqueID,sort=False):
        if 'GHCND_ID' in group.columns:
            if (len(group.GHCND_ID.unique())==1) & (group.GHCND_ID.astype(str).unique()[0]=='nan'):
                dfresult = dfresult.drop(group.index.tolist(),axis=0)
                continue
        #else:
        #print(name)
        #display(group[begin_column].unique())
        #display(group[end_column].unique())

        try:
            # 30075819 has a BEGIN_DATE of 10101 ??

            stn_start_date = pd.to_datetime(group.sort_values(begin_column)[begin_column].values[0],format='%Y%m%d')
            stn_end_date = pd.to_datetime(group.sort_values(end_column)[end_column].astype(str).str.replace('9999','2021').values[-1],format='%Y%m%d')
            #print(stn_start_date)
        except ValueError:
            continue
        if ((stn_start_date > pd.to_datetime(start_date)) & (stn_start_date < pd.to_datetime(end_date))) | (
             (stn_end_date > pd.to_datetime(start_date)) & (stn_end_date < pd.to_datetime(end_date))) | (
             (pd.to_datetime(start_date) > stn_start_date) & (pd.to_datetime(start_date) < stn_end_date)):
            # add number of relevant years in the range as an additional column
            #ndays = min([stn_end_date,pd.to_datetime(end_date)])- max([stn_start_date,pd.to_datetime(start_date)])
            ndays = pd.to_timedelta('0 days')
            for i,row in group.iterrows():

                ndays += max([pd.to_timedelta('0 days'),min([pd.to_datetime(str(row[end_column]).replace('9999','2021'),format='%Y%m%d'),pd.to_datetime(end_date)])-
                              max([pd.to_datetime(row[begin_column],format='%Y%m%d'),pd.to_datetime(start_date)])])

            #display(group.sort_values('BEGIN_DATE').head())
            #print(name,ndays.days)
            #display(group.head())
            # remove rows that don't have a GHCND_ID
            if 'GHCND_ID' in group.columns:
                dropindex = group[group.GHCND_ID.isnull()].index.tolist()
            else:
                dropindex=[]
            dfresult = dfresult.drop(dropindex,axis=0)
            # now remove duplicates based on group, keeping only one group member
            dropindex1 = [i for i in group.index if i not in dropindex]
            if len(dropindex1)> 1:
                dfresult = dfresult.drop(dropindex1[1:],axis=0)

            dfresult.loc[dfresult[uniqueID]==name,'NFRACTION'] = ndays.days / (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        else:
            dfresult = dfresult.drop(group.index.tolist(),axis=0)
            #display(dfresult)
    if 'NFRACTION' not in dfresult.columns:
        dfresult['NFRACTION'] = np.nan
    return dfresult
