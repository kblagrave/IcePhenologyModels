
from matplotlib import pyplot as plt

from numpy.polynomial import Chebyshev
import numpy as np

import pandas as pd

def iceon_off_summary(lkcode, year, df_lakeweather_offset_iceon,dfsmoothed,
                      df_zero_cross,date_range=(-140,198),
                      zc = True):
    # date_range is first and last day (relative to ice-on date)
    #   This was the range used to determine the zero-crossing.
    # zc only colours temperatures post zero-crossing (in top temperature time series bar plot)
    # THIS SUMMARY ONLY WORKS IF DURATION IS DEFINED IN df_zero_cross

    props = {'color':'1','alpha':0}
    dd_props = {'alpha':0.5,'facecolor':'1','edgecolor':'none'}

    nplots = 3
    top,bottom,left,right = 1,0.5,0.8,0.5
    hspace = 1.3
    figwidth = 183/25.4
    pltheight = 1
    pltratio = [2,1.5,1]
    avgpltheight = np.sum(pltratio)/ len(pltratio)
    pltwidth = figwidth - left - right
    figheight = bottom + top + pltheight*np.sum(pltratio) + hspace * (len(pltratio)-1)
    fig, axes = plt.subplots(nplots,1,sharex= False,
                         figsize = (figwidth,figheight),
                        gridspec_kw={'height_ratios': pltratio})

    fdd_color = 'C0'
    pdd_color = 'C3'
    nonfocus_color = '0.5'

    # get lake name from lakecode
    #lakename = dfice.loc[dfice.lakecode==lkcode,'lake'].drop_duplicates().values[0]

    dfoo = df_lakeweather_offset_iceon['TMINMAX'].copy()
    
        
    zero_cross = df_zero_cross.loc[(lkcode,year),:].copy()
    duration = zero_cross['ice_duration']
    ice_on_doy = zero_cross['ice_on_doy']
    #ice_off = zero_cross['ice_off_doy']
    zc_freeze = zero_cross['ZCFreezeDOY'] - ice_on_doy
    zc_thaw = zero_cross['ZCThawDOY'] - ice_on_doy
    lakename = zero_cross['lake'].title()

    smoothed_ts_doy = dfsmoothed.loc[(lkcode,year),:] # this is time series as day of year
    # shift smoothed_ts to be relative to ice_on_doy, like others
    print(ice_on_doy)
    smoothed_ts = smoothed_ts_doy.shift(-int(ice_on_doy))

    # set x limit for plotting
    if np.isnan(duration):
        xlim = (-100,200)
    else:
        xlim = (-100, duration +10)

    ####### TOP PLOT ######################
    #### Temperature time series ##########

    ax = axes.flatten()[0]
    ax2 = ax.twiny()
    ax2.set_xlabel("Time relative to ice on (days)")
    ax.set_xlabel("Time relative to ice off (days)")

    ax2.set_xlim(xlim)
    ax2.axvline(zc_freeze,lw=0.5, ls=':',color='0.5')
    ax2.axvline(zc_thaw,lw=0.5, ls=':',color='0.5')

    yorig = dfoo.loc[(lkcode,year),:].copy()

    y = yorig.copy()

    ind = ~y.isnull()
    y=y[ind]

    x = y.index.astype(float)
    y = y.astype(float).values
    
    if zc:
        ind  = (y < 0) & (x <= 0) & (x>=zc_freeze)
        ind2 = (y > 0) & (x > 0) & (x < duration) & (x >= zc_thaw)
    else:
        ind  = (y < 0) & (x <= 0)
        ind2 = (y > 0) & (x > 0) & (x < duration)

    ax2.bar(x[ind],np.abs(y[ind]),1,bottom=y[ind],color=fdd_color)
    ax2.bar(x[~ind & ~ind2],y[~ind & ~ind2],color=nonfocus_color)
    ax2.bar(x[ind2], y[ind2],color=pdd_color)

    #ax.plot(newx,newy)
    ax.set_ylabel('$T_\mathrm{air}$ (\N{DEGREE SIGN}C)')
    ax2.axhline(0,lw=0.5, color='k')
    ax2.axvline(0,lw=1,ls='-',color='k')

    yval = ax2.get_ylim()[1]/2.
    #ax2.text(0,yval,'ICE ON',weight='bold',rotation=90,ha='center',va='center', bbox=props)

    # now determine ice off axis based on duration
    ax.set_xlim(ax2.get_xlim()-duration)
    ax.axvline(0,lw=1,ls='-',color='k')
    yval = ax.get_ylim()[0]/2.
    #ax.text(0,yval,'ICE OFF',weight='bold',rotation=90,ha='center',va='center', bbox=props)


    #dfoo = dfoo.loc[:,-183:182] #.cumsum(axis=1)
    
    #c = Chebyshev.fit(x, y, deg=13)
    #newx = np.linspace(x[0],x[-1],1000)
    #newy = c(newx)
    
    ####   BOTTOM PLOT #######
    #### FDD and PDD   #######
    ax = axes.flatten()[2]
    ax2 = ax.twiny()

    ax2.set_xlabel("Time relative to ice on (days)")
    ax.set_xlabel("Time relative to ice off (days)")
    ax2.set_xlim(xlim)

    ## FDD
    y = yorig.copy()
    x = y.index.astype(float)
    if zc:
        ind = ~y.isnull() & (y<0) & (x >= zc_freeze)
    else:
        ind = ~y.isnull() & (y<0)
    y[~ind] = 0
    y=np.abs(y.cumsum())
    x = y.index.astype(float)
    y = y.astype(float).values
    ind = (x<= 0) & (y>0)
    ax2.fill_between(x[ind],y[ind], color=fdd_color, label="FDD")
    fdd = int(np.round(y[ind][-1],0))
    ax2.axvline(0,lw=1,ls='-',color='k')

    ## PDD
    y = yorig.copy()
    x = y.index.astype(float)
    if zc:
        ind = ~y.isnull() & (y>0) & (x >= zc_thaw)
    else:
        ind = ~y.isnull() & (y>0) & (x>0)
    y[~ind] = 0
    y=np.abs(y.cumsum())
    x = y.index.astype(float)
    y = y.astype(float).values
    ind = (y>0) & (x <=duration)
    ax2.fill_between(x[ind],y[ind],color=pdd_color, label='PDD')
    pdd = int(np.round(y[ind][-1],0))
    ax.set_xlim(ax2.get_xlim()-duration)
    ax.axvline(0,lw=1,ls='-',color='k')

    ax2.axvline(zc_freeze,lw=0.5,ls=':',color='0.5')
    ax2.axvline(zc_thaw, lw=0.5,ls=':',color='0.5')
    
    
    
    # add 10% to the yrange at the top
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    # set bottom of range to exactly 0
    ax.set_ylim(0, ax.get_ylim()[0] + yrange*1.1)
    yval = np.mean(ax2.get_ylim())
    ax2.text(zc_thaw,yval,f'{pdd}\npositive degree days\n(PDD)',rotation=0,ha='right',va='center',bbox=dd_props)
    ax2.annotate("", xy=(duration,pdd), xytext=(zc_thaw,yval),
             arrowprops=dict(arrowstyle="->"))
    yval = np.mean(ax2.get_ylim())
    ax2.text(zc_freeze,yval,f'{fdd}\nfreezing degree days\n(FDD)',rotation=0,ha='right',va='center', bbox=dd_props)
    ax2.annotate("", xy=(0,fdd), xytext=(zc_freeze,yval),
             arrowprops=dict(arrowstyle="->"))

    #ax.plot(x,y)
    ax.set_ylabel('Degree days (\N{DEGREE SIGN}C$\cdot$days)')
    #ax.set_xlabel('Days relative to ice event')

    #### MIDDLE PLOT ###
    # cumulative degree days for determining days after zero crossing
    ax = axes.flatten()[1]
    ax2= ax.twiny()
    
    # calculate cumulative sum the same way it was done elsewhere
    smoothed_range_doy = np.array([-114, 159])
    # convert to ice_on_doy
    smoothed_range = smoothed_range_doy - ice_on_doy

    ycumsum = yorig.loc[smoothed_range[0]:smoothed_range[1]].cumsum()
    y = ycumsum.astype(float).values
    x = ycumsum.index.astype(float) 

    ind = np.isnan(y) | np.isnan(x) | (x > date_range[1]) | (x < date_range[0])
    x = x[~ind]
    y = y[~ind]

    yr = y[(x>np.min(xlim)) & (x < np.max(xlim))]
    dyr = np.max(yr) - np.min(yr)
    yrange = np.min(yr) - dyr*0.05, np.max(yr) + dyr*0.05

    ax2.plot(x,y,lw=0.5,color='k')

    # smoothed 
    y = smoothed_ts.astype(float).values
    x = smoothed_ts.index.astype(float)
    ax2.plot(x,y,color="C2")

    #c = Chebyshev.fit(x, y, deg=13)
    #newx = np.linspace(x[0],x[-1],1000)
    #newy = c(newx)
    #maxy, maxy_ind = np.max(newy), np.argmax(newy)
    #miny, miny_ind = np.min(newy), np.argmin(newy)
    #ax2.plot(newx,newy, color="C0")
    #print(maxy, maxy_ind)
    
    arrow_width = 2
    arrow_length = 23
    arrow_style = f"simple,tail_width={arrow_width},head_length={arrow_width*0.5},head_width={arrow_width*1.5}"

    ax2.axvline(zc_freeze,lw=0.5,ls=':',color='0.5')
    ax2.annotate("", xy=(zc_freeze-arrow_length, np.mean(yrange)), xytext=(zc_freeze, np.mean(yrange)),
             arrowprops=dict(arrowstyle=arrow_style,facecolor='C3', alpha=0.2, edgecolor='none'))
    ax2.annotate("", xy=(zc_freeze+arrow_length, np.mean(yrange)), xytext=(zc_freeze, np.mean(yrange)),
             arrowprops=dict(arrowstyle=arrow_style,facecolor='C0', alpha=0.2, edgecolor='none'))
   
    ax2.text(zc_freeze-2, np.mean(yrange),"$T > 0$\N{DEGREE SIGN}C", fontsize=10, ha='right',va='center')
    #ax2.text(zc_freeze, np.mean(yrange), '$T = 0$\N{DEGREE SIGN}C', fontsize=8, rotation= 0, ha='center',va='center')
    ax2.text(zc_freeze+2, np.mean(yrange),"$T < 0$\N{DEGREE SIGN}C", fontsize=10, ha='left',va='center')

    ax2.annotate("", xy=(zc_thaw-arrow_length, np.mean(yrange)), xytext=(zc_thaw, np.mean(yrange)),
             arrowprops=dict(arrowstyle=arrow_style,facecolor='C0', alpha=0.2,edgecolor='none'))
    ax2.annotate("", xy=(zc_thaw+arrow_length, np.mean(yrange)), xytext=(zc_thaw, np.mean(yrange)),
             arrowprops=dict(arrowstyle=arrow_style,facecolor='C3', alpha=0.2, edgecolor='none'))

    ax2.axvline(zc_thaw, lw=0.5,ls=':',color='0.5')
    ax2.text(zc_thaw-2, np.mean(yrange),"$T < 0$\N{DEGREE SIGN}C", fontsize=10, ha='right',va='center')
    #ax2.text(zc_thaw, np.mean(yrange), '$T = 0$\N{DEGREE SIGN}C', fontsize=8, rotation= 0, ha='center',va='center')
    ax2.text(zc_thaw+2, np.mean(yrange),"$T > 0$\N{DEGREE SIGN}C", fontsize=10, ha='left',va='center')

    
    #ax2.axvline(newx[maxy_ind],lw=0.5,ls=':',color='0.5')
    #ax2.text(newx[maxy_ind], np.mean(yr), 'ZERO CROSSING', rotation= 90, ha='center',va='center')
    #ax2.axvline(newx[miny_ind], lw=0.5,ls=':',color='0.5')
    #ax2.text(newx[miny_ind], np.mean(yr), 'ZERO CROSSING', rotation = 90, ha='center',va='center')
    
    ax2.set_xlabel("Time relative to ice on (days)")
    ax.set_xlabel("Time relative to ice off (days)")

    ax.set_ylabel("Degree days (\N{DEGREE SIGN}C$\cdot$days)")

    ax2.set_ylim(yrange)
    ax2.set_xlim(xlim)
    # now determine ice off axis based on duration
    ax.set_xlim(ax2.get_xlim()-duration)

    ax2.axvline(0,lw=1,ls='-',color='k')
    yval = np.mean(ax2.get_ylim())
    #ax2.text(0,yval,'ICE ON',weight='bold', rotation=90,ha='center',va='center', bbox=props)

    ax.axvline(0,lw=1,ls='-',color='k')
    yval = np.mean(ax.get_ylim())
    #ax.text(0,yval,'ICE OFF',weight='bold', rotation=90,ha='center',va='center', bbox=props)

    fig.text(0.01,(bottom + pltheight*pltratio[2]+hspace*0.3)/figheight,'(c)',weight='bold')
    fig.text(0.01,(bottom + pltheight*(pltratio[1]+pltratio[2])+hspace + hspace*0.3)/figheight,'(b)',weight='bold')

    fig.text(0.01,(bottom + pltheight*(pltratio[1]+pltratio[2]+pltratio[0])+hspace*2 + hspace*0.3)/figheight,'(a)',weight='bold')


    fig.subplots_adjust(left = left/figwidth, right=1 - right/figwidth,
                        top = 1 - top/figheight, bottom = bottom/figheight,
                        hspace = hspace/avgpltheight)
    fig.text(0.06,0.98,f"{lakename} {year}-{year+1}", weight='bold',fontsize=12,
             ha='left',va='top',transform=fig.transFigure)
    if zc:
        fig.savefig(f'{lkcode}_{year}_ZC.pdf')
        fig.savefig(f'{lkcode}_{year}_ZC.png',dpi=300)
    else:
        fig.savefig(f'{lkcode}_{year}.pdf')
        fig.savefig(f'{lkcode}_{year}.png',dpi=300)

    #ax.set_ylim(1500,2500)