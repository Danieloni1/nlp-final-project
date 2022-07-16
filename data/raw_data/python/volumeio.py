import h5py

from bvframework.datamodel.pipeline import Volume


def read_volume(volume_path):
    info = {}
    vol = None
    with h5py.File(volume_path, 'r') as f:
        dset = f['volume']
        vol = dset.value

        for item in dset.attrs.keys():
            info[item] = dset.attrs[item]

    return Volume(vol, info)


def read_volume_info(volume_path):
    info = {}
    with h5py.File(volume_path, 'r') as f:
        dset = f['volume']
        for item in dset.attrs.keys():
            info[item] = dset.attrs[item]

    return info


def save_volume(volume_path, volume, compression=None):
    f = h5py.File(volume_path, "w")
    dset = f.create_dataset("volume", data=volume.data, compression=compression)  # compression="gzip"

    if volume.info is not None:
        for key in volume.info:
            dset.attrs[key] = volume.info[key]

    f.close()
