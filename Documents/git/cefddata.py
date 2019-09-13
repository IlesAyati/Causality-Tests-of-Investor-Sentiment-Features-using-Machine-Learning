# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:07:03 2019

@author: yeeya
"""
import wrds
db           = wrds.Connection(wrds_username='iayati')
#db.create_pgpass_file()
#import psycopg2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Listing libraries and tables (just to have an idea on how the datasets look)
lib         = conn.list_libraries()
listtble    = conn.list_tables(library="crsp")
descrtble   = conn.describe_table(library="crsp",table="msf")
descrtble2  = conn.describe_table(library="comp",table="secm")

# =============================================================================
# Extracting all necessary data at once, instead of creating tables and
# referring to them to create new tables. 
# The first 4 lines define which columns are extracted from the four tables
# tables crsp.msf (a), crsp.ccmxpf_linktable (b), crsp msenames (c) and 
# comp.secm (d). 
# 
# a,b and c extract the prices, returns etc. and match with identifiers of the closed-end funds, 
# where as d extracts the navs and merges them with the appropriate funds. 
# d also creates a column for discounts (price-nav)/nav.
# =============================================================================


data           = db.raw_sql(""" 
                            select a.permno, a.date, c.shrcd,
                            a.ret, a.vol, a.shrout, a.prc, b.gvkey, b.liid,
                            b.lpermno, d.navm, d.datadate, d.iid, d.gvkey,
                            (abs(a.prc)-d.navm)/d.navm as discount
                            from crsp.msf as a
                            left join crsp.ccmxpf_linktable as b 
                            on a.permno = b.lpermno
                            and (a.date >= b.linkdt)
                            and (a.date <= b.linkenddt)
                            and b.linktype in ('LC','LU')
                            left join crsp.msenames as c
                            on a.permno=c.permno
                            left join comp.secm as d
                            on b.gvkey=d.gvkey
                            and b.liid=d.iid
                            and (a.date = d.datadate)
                            where a.date between '01/01/1998' and '12/31/2018'
                            and c.shrcd=14
                            """)

db.close()

data.shape
print(data.columns)
# Convert Data to array
npdata      = np.array(data)
#Count number of real values in the discounts column (non-NaNs)
nzdata      = np.count_nonzero(~pd.isnull(npdata[:,-1]))
# 87791 rows out of 339205 ~ 1/4 of the dataset.
# Let's find the index of the entries that are not NaNs
nonNaN = np.nonzero(~pd.isnull(npdata[:,-1]))
# and the date?
firstnonNaN = npdata[11749,:]


