# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:46:35 2019

@author: yeeya
"""
import pandas as pd
import wrds
import psycopg2
conn = wrds.Connection()
conn.create_pgpass_file()

lib         = conn.list_libraries()
listtble    = conn.list_tables(library="crsp")
descrtble   = conn.describe_table(library="crsp",table="msf")
descrtble2  = conn.describe_table(library="crsp",table="ccmxpf_linktable")

# Here I retrieve the monthly securities data conditioned on share code = 14, 
# which is the code for Closed-End funds:
crsp_m      = conn.raw_sql("""
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
compstat    = conn.raw_sql(""" 
                            select a.*, b.gvkey, b.liid
                            from crsp.msf as a, crsp.ccmxpf_linktable as b 
                            where
                            a.permno = b.lpermno
                            and (a.date >= b.linkdt)
                            and (a.date <= b.linkenddt)
                            and b.linktype in ('LC','LU')
                            and a.date between '01/01/1998' and '12/31/2018'
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
navfile    = conn.raw_sql("""
                            select a.*, b.navm, (abs(a.prc)-b.navm)/b.navm as discount
                            from crsp.msf as a, comp.secm as b
                            where
                            a.gvkey=b.gvkey
                            and a.liid=b.iid
                            and intnx( 'month' ,a.date, 0 , 'E' ) = b.datadate
                            """)
conn.close()
