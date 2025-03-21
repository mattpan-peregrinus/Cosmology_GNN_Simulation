import h5py

filename="/Users/matthewpan/Desktop/fullrun.hdf5"
with h5py.File(filename, "r") as f:
    print(f.keys())

    coords = f['Coordinates'][()]
    accs = f['HydroAcceleration'][()]
    vels = f['Velocities'][()]

    print(f'coords.shape = {coords.shape}')
    print(f'accs.shape = {accs.shape}')
    print(f'vels.shape = {vels.shape}')
