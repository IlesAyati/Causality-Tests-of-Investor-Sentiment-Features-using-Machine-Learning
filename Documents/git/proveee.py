# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:46:35 2019

@author: yeeya
"""

import wrds
db           = wrds.Connection(wrds_username='iayati')
import pandas as pd
import numpy as np
import psycopg2

#conn.create_pgpass_file()


lib         = conn.list_libraries()
listtble    = conn.list_tables(library="crsp")
descrtble   = conn.describe_table(library="crsp",table="msf")
descrtble2  = conn.describe_table(library="comp",table="secm")

# Here I retrieve the monthly securities data conditioned on share code = 14, 
# which is the code for Closed-End funds:
crsp_m      = db.raw_sql("""
                          select a.permno, a.permco, b.ncusip, a.date, 
                          b.shrcd, b.exchcd, b.siccd,
                          a.ret, a.vol, a.shrout, a.prc
                          from crsp.msf as a
                          left join crsp.msenames as b
                          on a.permno=b.permno
                          and b.namedt<=a.date
                          and a.date<=b.nameendt
                          where a.date between '01/01/1998' and '12/31/2018'
                          and b.shrcd=14
                          """) 
# change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# Here I link the same data as I retrieved in crsp_m with Compustat identifiers. 
# Note that I cannot refer to crsp_m as it is not a table. 
# Instead I have to match with the server table crsp.msf as in lines 20-31 again:


## What if I do everything in one step?
data           = db.raw_sql(""" 
                            select a.permno, a.permco, a.date, c.shrcd,
                            a.ret, a.vol, a.shrout, a.prc, b.gvkey, b.liid, 
                            d.navm, d.datadate, d.iid, d.gvkey,
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

# Still I can merge these two datasets together on permno (unique permanent security identification number)
# and date:
crspp = pd.merge(crsp_m, compstat, how='left',on=['permno','date'])

# Then ideally, I would merge that dataframe (crspp) with data on net asset values (nav).. the diff
# between prices (from crsp.msf) and navs is the Closed-End fund discount, 
# which is the measure I want to use..
# This is not feasible because I have to match gvkeys (company code). This would be possible if 
# crspp was a created table, but it's not. So I have to find a way to match these without
# referring to the server table crsp.msf (which does not contain gvkeys).
#
navfile    = db.raw_sql("""
                            select navm
                            from comp.secm
                            where
                            datadate between '01/01/1998' and '12/31/2018'
                            """)
conn.close()
