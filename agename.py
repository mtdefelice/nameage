import urllib.request
import os
import glob
import zipfile

import numpy as np, pandas as pd

import matplotlib.pyplot as plt
plt.style.use ('ggplot')

# Function to generate nameage plots
def gen_na_plots (df, _nam = 'Rebecca', _sex = 'F'):
    dm = pd.read_csv ('~/ds/DeathProbsE_{}_Alt2_TR2014.csv'.format (_sex), skiprows = 1)
    a = dm.set_index ('Year').loc[pd.to_datetime ('now').year].rename ('pd')
    a.index = a.index.astype ('int64')
    a.index.name = 'age'

    b = df[(df.sex == _sex) & (df.name == _nam)].groupby (['age']).n.sum ()

    c = pd.concat ([
        b,
        a,
    ], axis = 1).dropna ()

    c['pl'] = 1 - c.pd
    c['plc'] = c.pl.cumprod ()
    c['adj'] = (c.n * c.plc)
    c['by'] = pd.to_datetime ('now').year - c.index
    c['adjc'] = c.adj.cumsum ()

    d = c[['n', 'adj']]
    d.index.name = ''
    d.rename (columns = {'n': 'Born', 'adj': 'Expected Living (2017)'}).plot (title = '{}: US Age Distribution'.format (_nam))
    plt.axvline (x = c[c.adjc >= c.adjc.mean ()].iloc[0].name, c = 'k', ls = ':', lw = 0.8)
    plt.savefig ('an-{}.pdf'.format (_nam.lower ()))
    
    # Another way to visualize
    '''
    d = c.set_index (['by'])[['n', 'adj']]
    d.index.name = ''
    d.rename (columns = {'n': 'Born', 'adj': 'Expected Living (2017)'}).plot (title = '{}: US SSN Registrations by Birth Year'.format (_nam))
    plt.axvline (x = pd.to_datetime ('now').year - c[c.adjc >= c.adjc.mean ()].iloc[0].name, c = 'k', ls = ':', lw = 0.8)
    plt.savefig ('an-{}_alt.pdf'.format (_nam.lower ()))
    '''

if __name__ == '__main__':
    # Download data sources from ssa.gov
    for _ in [
        'https://www.ssa.gov/oact/babynames/state/namesbystate.zip',
        'https://www.ssa.gov/oact/HistEst/Death/2014/DeathProbsE_F_Alt2_TR2014.csv',
        'https://www.ssa.gov/oact/HistEst/Death/2014/DeathProbsE_M_Alt2_TR2014.csv',
    ]:
        if not os.path.isfile ('ds/{}'.format (os.path.basename (_))):
            with urllib.request.urlopen (_) as p, open ('ds/{}'.format (os.path.basename (_)), 'wb') as f:
                f.write (p.read ())

    # Unzip
    for _ in glob.glob (r'ds/*.zip'):
        with zipfile.ZipFile (_, 'r') as z:
            z.extractall ('ds/namesbystate')

    # Create a pandas DataFrame; combine each state's birth records
    df = pd.DataFrame ()
    for _ in glob.glob ('ds/namesbystate/*.TXT'):
        print ('Adding: {}'.format (_))
        a = pd.read_csv (_, header = None, names = ['state', 'sex', 'by', 'name', 'n'])
        df = df.append (a)

    df['age'] = pd.to_datetime ('now').year - df.by

    # Generate nameage plots
    gen_na_plots (df, _nam = 'John', _sex = 'M')
    gen_na_plots (df, _nam = 'Paul', _sex = 'M')
    gen_na_plots (df, _nam = 'Lucy', _sex = 'F')
    gen_na_plots (df, _nam = 'Rita', _sex = 'F')

