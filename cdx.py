import cdx_toolkit

url = 'https://medium.com/*'
cdx = cdx_toolkit.CDXFetcher(source='cc')

objs = list(cdx.iter(url, from_ts='202002', to='202006', limit=5000, filter='=status:200'))

print(len(objs))

for o in objs[:10]:
    print(o.data) 
