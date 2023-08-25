def split_field(data:str):
    sep=r'~'
    if data.find(sep)>0:
        ds=data.split(sep)
        return list(range( int(ds[0]),int(ds[-1])+1 ))
    sep=r'|'
    if data.find(sep)>0:
        ds=data.split(sep)
        return list(map(int,ds))
    return [float(data)]