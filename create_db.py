import re
import gzip
import sqlite3
from sqlite3 import Error
from os import listdir
from os.path import isfile, join

proc_path = './processed_wet/'
hsfiles = [join(proc_path,f) for f in listdir(proc_path) if f[-5:] == "hs.gz"]

print(hsfiles)

def dequotify(s):
    s = s.replace("'","&#39;")
    s = s.replace('"',"&#34;")
    return s

def sql_connection():
    try:
        con = sqlite3.connect('quadrants.db')
        print("Connection is established: Database is created in memory")
    except Error:
        print(Error)
    return con

def sql_quadrant_table(con, quadrant, urls):
    cursor = con.cursor()
    cursor.execute("CREATE TABLE Q"+quadrant+"(id integer PRIMARY KEY, url text, keywords text)")
    con.commit()

    ID=0
    for u,k in urls.items():
        ID+=1
        url=u
        keywords=k
        print("INSERT INTO Q"+quadrant+" VALUES('"+str(ID)+"','"+dequotify(url)+"','"+keywords+"')")
        cursor.execute("INSERT INTO Q"+quadrant+" VALUES('"+str(ID)+"','"+dequotify(url)+"','"+keywords+"')")
    con.commit()

quadrants = {}
con = sql_connection()

for filename in hsfiles:
    with gzip.open(filename,'r') as f:
        for l in f:
            l = l.decode("utf-8").rstrip('\n')
            fields = l.split()
            hs = fields[0]
            if hs not in quadrants:
                quadrants[hs] = {}
            url = fields[1]
            keywords = fields[2:]
            keywords = ' '.join([dequotify(k) for k in keywords])
            quadrants[hs][url] = keywords
        f.close()

for quadrant, urls in quadrants.items():
    sql_quadrant_table(con, quadrant, urls)

con.close()

