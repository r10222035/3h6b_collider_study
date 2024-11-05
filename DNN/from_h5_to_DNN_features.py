import sys
import h5py
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from scipy.stats import skew
from itertools import repeat

N_CORES = 64


def DeltaR(eta1, phi1, eta2, phi2):
    dEta = eta1 - eta2
    dPhi = np.abs(phi1 - phi2)
    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)
    dR = np.sqrt(dPhi**2 + dEta**2)
    return dR


def DeltaA(eta1, phi1, eta2, phi2):
    dEta = eta1 - eta2
    dPhi = np.abs(phi1 - phi2)
    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)
    dA = np.cosh(dEta) - np.cos(dPhi)
    return dA


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
    eta = 1 / 2 * np.log((P + pz) / (P - pz))
    phi = np.arctan(py / px)
    m = np.sqrt(e**2 - px**2 - py**2 - pz**2)

    return pt, eta, phi, m


def construct_inputs(h5_file, start, end):

    with h5py.File(h5_file, 'r') as f:

        dR = [[], [], []]
        rms_dR = []
        dA_skew = []
        HT = []
        mhCostheta = []
        eta_mhhh_fraction = []
        sphericity = []
        aplanarity = []

        for event in tqdm(range(start, end)):

            nj = f['INPUTS/Source/MASK'][event].sum()
            pt = f['INPUTS/Source/pt'][event]
            eta = f['INPUTS/Source/eta'][event]
            phi = f['INPUTS/Source/phi'][event]
            mass = f['INPUTS/Source/mass'][event]
            btag = f['INPUTS/Source/btag'][event]

            # for pairing
            jets_index = np.where(btag)[0][0:6]

            h1b1 = f['TARGETS/h1/b1'][event]
            h1b2 = f['TARGETS/h1/b2'][event]
            h2b1 = f['TARGETS/h2/b1'][event]
            h2b2 = f['TARGETS/h2/b2'][event]
            h3b1 = f['TARGETS/h3/b1'][event]
            h3b2 = f['TARGETS/h3/b2'][event]

            dR1 = DeltaR(eta[h1b1], phi[h1b1], eta[h1b2], phi[h1b2])
            dR2 = DeltaR(eta[h2b1], phi[h2b1], eta[h2b2], phi[h2b2])
            dR3 = DeltaR(eta[h3b1], phi[h3b1], eta[h3b2], phi[h3b2])

            dR[0].append(dR1)
            dR[1].append(dR2)
            dR[2].append(dR3)

            # compute rms of dR, consider all possible combinations
            dR_dijets = [DeltaR(eta[i], phi[i], eta[j], phi[j]) for i in jets_index for j in jets_index if i < j]
            rms_dR.append(np.sqrt(np.mean(np.square(dR_dijets))))

            # compute skewness of dA
            dA_dijets = [DeltaA(eta[i], phi[i], eta[j], phi[j]) for i in jets_index for j in jets_index if i < j]
            dA_skew.append(skew(dA_dijets))

            jets = np.array([[pt[i], eta[i], phi[i], mass[i]] for i in [h1b1, h1b2, h2b1, h2b2, h3b1, h3b2]])

            _, _, _, mh1 = PtEtaPhiM(*PxPyPzE(jets[[0, 1]]))
            _, _, _, mh2 = PtEtaPhiM(*PxPyPzE(jets[[2, 3]]))
            _, _, _, mh3 = PtEtaPhiM(*PxPyPzE(jets[[4, 5]]))

            HT.append(jets[:, 0].sum())

            mh_ref = np.array([120, 115, 110])
            mh_rec = np.array([mh1, mh2, mh3]) - mh_ref

            mhCostheta.append(mh_rec.dot(mh_ref) / (np.linalg.norm(mh_rec) * np.linalg.norm(mh_ref)))

            # eta - mhhh fraction
            _, _, _, mhhh = PtEtaPhiM(*PxPyPzE(jets))

            tmp = 0
            for i in jets_index:
                for j in jets_index:
                    if i < j:
                        tmp += 2 * pt[i] * pt[j] * (np.cosh(eta[i] - eta[j]) - 1)

            eta_mhhh_fraction.append(tmp / mhhh**2)

            # Sphericity and Aplanarity
            Mxyz = np.zeros((3, 3))
            p_total = 0
            for i in range(6):
                px, py, pz, _ = PxPyPzE(jets[i])

                Mxyz += np.outer([px, py, pz], [px, py, pz])

                p_total += (px**2 + py**2 + pz**2)

            Mxyz /= p_total
            eigvals = np.linalg.eigvals(Mxyz)
            eigvals = np.sort(eigvals)[::-1]

            sphericity.append(3 / 2 * (eigvals[1] + eigvals[2]))
            aplanarity.append(3 / 2 * eigvals[2])

        # save the features to npy file
        results = np.array([dR[0], dR[1], dR[2], rms_dR, dA_skew, HT, mhCostheta, eta_mhhh_fraction, sphericity, aplanarity]).transpose()
        return results
    

def from_h5_to_DNN_feature(h5_file, output_file):
    # file_path: input h5 file path
    # output_path: output h5 file path with jet pairing results
    # use_btag: whether to use btag information


    with h5py.File(h5_file, 'r') as f:
        nevent = f['INPUTS/Source/pt'].shape[0]
    print(f'Number of events: {nevent}')

    # Multi-core processing
    print(f'Number of cores: {N_CORES}')
    start = [nevent // N_CORES * i for i in range(N_CORES)]
    end = [nevent // N_CORES * (i+1) for i in range(N_CORES)]
    end[-1] = nevent

    with mp.Pool(processes=N_CORES) as pool:
        results = pool.starmap(construct_inputs, zip(repeat(h5_file), start, end))

    data = np.concatenate(results)
    np.save(output_file, data)


if __name__ == '__main__':

    h5_file = sys.argv[1]
    output_file = sys.argv[2]
    from_h5_to_DNN_feature(h5_file, output_file)
