import os
import sys
import h5py
import shutil
import itertools

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from itertools import repeat

N_CORES = 64


def all_pairs(lst):
    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in all_pairs(lst[:i] + lst[i+1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1, len(lst)):
            pair = (a,lst[i])
            for rest in all_pairs(lst[1:i]+lst[i+1:]):
                yield [pair] + rest

                
def Mjets(jets):
    # jets: 一個形狀為 (n, 4) 的 NumPy 陣列，其中 n 是噴射數量，每個噴射有四個屬性（pt, eta, phi, m）

    pt, eta, phi, m = jets.T  # 將噴射屬性分解為單獨的陣列

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(m*m + px*px + py*py + pz*pz)

    return np.sqrt(e.sum()**2 - px.sum()**2 - py.sum()**2 - pz.sum()**2)


def PxPyPzE(jets):
    # jets: 一個形狀為 (n, 4) 的 NumPy 陣列，其中 n 是噴射數量，每個噴射有四個屬性（pt, eta, phi, m）
    pt, eta, phi, m = jets.T

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(m*m + px*px + py*py + pz*pz)

    return px.sum(), py.sum(), pz.sum(), e.sum()


def PtEtaPhiM(px, py, pz, e):

    P = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    eta = 1/2 * np.log((P + pz)/(P - pz))
    phi = np.arctan(py/px)
    m = np.sqrt(e**2 - px**2 - py**2 - pz**2)

    return pt, eta, phi, m


def chi2_triHiggs(m1, m2, m3):
    mh = 125.0
    return (m1 - mh)**2 + (m2 - mh)**2 + (m3 - mh)**2


def abs_triHiggs(m1, m2, m3):
    return abs(m1 - 120) + abs(m2 - 115) + abs(m3 - 110)


def write_pairing_results(file, data: list):
    pairing = np.array(data).transpose()

    # Write
    file['TARGETS/h1/b1'][:] = pairing[0]
    file['TARGETS/h1/b2'][:] = pairing[1]
    file['TARGETS/h2/b1'][:] = pairing[2]
    file['TARGETS/h2/b2'][:] = pairing[3]
    file['TARGETS/h3/b1'][:] = pairing[4]
    file['TARGETS/h3/b2'][:] = pairing[5]


def pair_jets(file_path, pairing_method, start, end):

    with h5py.File(file_path, 'r') as f:
        pairing_list = []
        for event in tqdm(range(start, end)):

            nj = f['INPUTS/Source/MASK'][event].sum()
            pt = f['INPUTS/Source/pt'][event]
            eta = f['INPUTS/Source/eta'][event]
            phi = f['INPUTS/Source/phi'][event]
            mass = f['INPUTS/Source/mass'][event]
            btag = f['INPUTS/Source/btag'][event]

            chisq = -1 
            pair = []

            nbj = np.sum(btag)
            if nbj == 4:
                jets_index = np.concatenate([np.where(btag)[0][0:4], np.where(~btag)[0][0:2]])
            elif nbj == 5:
                jets_index = np.concatenate([np.where(btag)[0][0:5], np.where(~btag)[0][0:1]])
            elif nbj == 6:
                jets_index = np.where(btag)[0][0:6]

            for combination in itertools.combinations(jets_index, 6):
                for (i1,i2), (i3,i4), (i5,i6) in all_pairs(combination):       
                    jets = np.array([[pt[i], eta[i], phi[i], mass[i]] for i in [i1, i2, i3, i4, i5, i6]])
            
                    pt1, _, _, mh1 = PtEtaPhiM(*PxPyPzE(jets[[0, 1]]))
                    pt2, _, _, mh2 = PtEtaPhiM(*PxPyPzE(jets[[2, 3]]))
                    pt3, _, _, mh3 = PtEtaPhiM(*PxPyPzE(jets[[4, 5]]))

                    pt_mh_pairs = sorted(zip([pt1, pt2, pt3], [mh1, mh2, mh3], [(i1, i2), (i3, i4), (i5, i6)]))
                    pt_sorted, mh_sorted, pair_sorted = zip(*pt_mh_pairs)

                    mh1, mh2, mh3 = mh_sorted[::-1]
                    tem = pairing_method(mh1, mh2, mh3)

                    if chisq < 0 or tem < chisq:
                        chisq = tem
                        pair = [jet for pair in pair_sorted[::-1] for jet in pair]

            pairing_list.append(pair)

        return pairing_list
    

def perform_jet_pairing(file_path, output_path, pairing_method=chi2_triHiggs):
    # file_path: input h5 file path
    # output_path: output h5 file path with jet pairing results
    # use_btag: whether to use btag information

    shutil.copy(file_path, output_path)

    with h5py.File(file_path, 'r') as f:
        nevent = f['INPUTS/Source/pt'].shape[0]
    print(f'Number of events: {nevent}')

    # Multi-core processing
    print(f'Number of cores: {N_CORES}')
    start = [nevent // N_CORES * i for i in range(N_CORES)]
    end = [nevent // N_CORES * (i+1) for i in range(N_CORES)]
    end[-1] = nevent

    with mp.Pool(processes=N_CORES) as pool:
        results = pool.starmap(pair_jets, zip(repeat(file_path), repeat(pairing_method), start, end))

    pairing_list = [pairing for result_list in results for pairing in result_list]

    # write to h5 file
    with h5py.File(output_path, 'a') as f_out:
        write_pairing_results(f_out, pairing_list)


if __name__ == '__main__':

    file_path = sys.argv[1]
    output_path = sys.argv[2]
    if sys.argv[3] == 'chi2_triHiggs':
        pairing_method = chi2_triHiggs
    elif sys.argv[3] == 'abs_triHiggs':
        pairing_method = abs_triHiggs
    else:
        print('Invalid pairing method.')
        sys.exit(1)

    perform_jet_pairing(file_path, output_path, pairing_method)