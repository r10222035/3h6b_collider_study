#!/usr/bin/env python
# coding: utf-8
# generate tri-Higgs HDF5 data for SPANet
# jets are required PT > 25 GeV
# with correct jet assignment
# python from_root_to_h5.py <root file path> <output file name> <minimum b-jet>

import sys
import h5py
import ROOT

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from itertools import repeat

ROOT.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/')
ROOT.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/external/')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootConfReader.h"')
ROOT.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTask.h"')
ROOT.gSystem.Load("/usr/local/Delphes-3.4.2/install/lib/libDelphes")

MAX_JETS = 15
N_CORES = 16


def DeltaR(eta1, phi1, eta2, phi2):
    dEta = eta1 - eta2
    dPhi = abs(phi1 - phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5

    return dR


def create_triHiggs_dataset(f, nevent, MAX_JETS):
    # with b-tagging information
    f.create_dataset('INPUTS/Source/MASK', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')
    f.create_dataset('INPUTS/Source/pt', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('INPUTS/Source/eta', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('INPUTS/Source/phi', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('INPUTS/Source/mass', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('INPUTS/Source/btag', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')

    f.create_dataset('TARGETS/h1/b1', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('TARGETS/h1/b2', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('TARGETS/h2/b1', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('TARGETS/h2/b2', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('TARGETS/h3/b1', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('TARGETS/h3/b2', (nevent,), maxshape=(None,), dtype='<i8')

    f.create_dataset('CLASSIFICATIONS/EVENT/signal', (nevent,), maxshape=(None,), dtype='<i8')


def get_particle_mask(quarks_Jet, quarks_index):
    # quarks_index: 粒子對應的夸克編號
    # 若某粒子的 每個夸克都有對應到 jet 且 每個夸克對應的 jet 都沒有重複，則返回 true
    mask = True
    for i in quarks_index:
        if quarks_Jet[i] == -1:
            mask = False
        else:
            for j in range(len(quarks_Jet)):
                if j == i:
                    continue
                if quarks_Jet[i] == quarks_Jet[j]:
                    mask = False
    return mask


def write_dataset(file, data: list):
    nevent = len(data)

    for key in data[0].keys():
        # Resize
        shape = list(file[key].shape)
        shape[0] = nevent
        file[key].resize(shape)
        # Write
        value = np.array([data_dict[key] for data_dict in data])
        file[key][:] = value


def get_dataset_keys(f):
    # 取得所有 Dataset 的名稱
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def select_event(root_path, nbj_min, start, end):

    f = ROOT.TFile(root_path)
    tree = f.Get("Delphes")

    data_list = []
    for i in tqdm(range(start, end)):
        tree.GetEntry(i)

        # 夸克資料
        # b夸克 衰變前的編號
        quarks_id = []
        quarks_Eta = []
        quarks_Phi = []
        quarks_Jet = [-1, -1, -1, -1, -1, -1]

        # 找出 3 個 final Higgs
        final_h_index = []
        for index, particle in enumerate(tree.Particle):
            if particle.PID == 25:
                h = index
                d1 = tree.Particle[h].D1
                while tree.Particle[d1].PID == 25:
                    h = d1
                    d1 = tree.Particle[h].D1
                final_h_index.append(h)

        final_h_index = list(set(final_h_index))

        # 找出 6個 final b quark
        for h in final_h_index:
            # h > b b~
            b1 = tree.Particle[h].D1
            b2 = tree.Particle[h].D2

            # 找出 b 衰變前的編號
            d1 = tree.Particle[b1].D1
            while abs(tree.Particle[d1].PID) == 5:
                b1 = d1
                d1 = tree.Particle[b1].D1

            # 找出 b~ 衰變前的編號
            d2 = tree.Particle[b2].D1
            while abs(tree.Particle[d2].PID) == 5:
                b2 = d2
                d2 = tree.Particle[b2].D1

            quarks_id.extend([b1, b2])

        quarks_Eta.extend(tree.Particle[quark].Eta for quark in quarks_id)
        quarks_Phi.extend(tree.Particle[quark].Phi for quark in quarks_id)

        # 事件中的 jet 資料
        jet_PT = np.array([jet.PT for jet in tree.Jet])
        jet_Eta = np.array([jet.Eta for jet in tree.Jet])
        jet_Phi = np.array([jet.Phi for jet in tree.Jet])
        jet_Mass = np.array([jet.Mass for jet in tree.Jet])
        jet_BTag = np.array([jet.BTag for jet in tree.Jet])

        # Jet 資料
        # |eta| < 2.5 & PT > 25 GeV
        eta_pt_cut = np.array((np.abs(jet_Eta) < 2.5) & (jet_PT > 25))

        nj = eta_pt_cut.sum()

        # 至少要 6 jet
        if nj < 6:
            continue

        nbj = np.array(jet_BTag[eta_pt_cut][:MAX_JETS]).sum()
        # 在前 MAX_JETS jets 中，至少要 nbj_min 個 b-jet
        if nbj < nbj_min:
            continue

        PT = np.array(jet_PT[eta_pt_cut])
        Eta = np.array(jet_Eta[eta_pt_cut])
        Phi = np.array(jet_Phi[eta_pt_cut])
        Mass = np.array(jet_Mass[eta_pt_cut])
        BTag = np.array(jet_BTag[eta_pt_cut])

        # 找出每個夸克配對的 jet
        more_than_1_jet = False
        for quark in range(len(quarks_Jet)):
            for i in range(min(nj, MAX_JETS)):
                dR = DeltaR(Eta[i], Phi[i], quarks_Eta[quark], quarks_Phi[quark])
                if dR < 0.4 and quarks_Jet[quark] == -1:
                    quarks_Jet[quark] = i
                elif dR < 0.4:
                    more_than_1_jet = True

        if more_than_1_jet:
            continue

        h1_mask = get_particle_mask(quarks_Jet, quarks_index=(0, 1))
        h2_mask = get_particle_mask(quarks_Jet, quarks_index=(2, 3))
        h3_mask = get_particle_mask(quarks_Jet, quarks_index=(4, 5))

        # 準備寫入資料
        data_dict = {
            'INPUTS/Source/MASK': np.arange(MAX_JETS) < nj,
            'INPUTS/Source/pt': PT[:MAX_JETS] if nj > MAX_JETS else np.pad(PT, (0, MAX_JETS-nj)),
            'INPUTS/Source/eta': Eta[:MAX_JETS] if nj > MAX_JETS else np.pad(Eta, (0, MAX_JETS-nj)),
            'INPUTS/Source/phi': Phi[:MAX_JETS] if nj > MAX_JETS else np.pad(Phi, (0, MAX_JETS-nj)),
            'INPUTS/Source/mass': Mass[:MAX_JETS] if nj > MAX_JETS else np.pad(Mass, (0, MAX_JETS-nj)),
            'INPUTS/Source/btag': BTag[:MAX_JETS] if nj > MAX_JETS else np.pad(BTag, (0, MAX_JETS-nj)),

            'TARGETS/h1/b1': quarks_Jet[0] if h1_mask else -1,
            'TARGETS/h1/b2': quarks_Jet[1] if h1_mask else -1,
            'TARGETS/h2/b1': quarks_Jet[2] if h2_mask else -1,
            'TARGETS/h2/b2': quarks_Jet[3] if h2_mask else -1,
            'TARGETS/h3/b1': quarks_Jet[4] if h3_mask else -1,
            'TARGETS/h3/b2': quarks_Jet[5] if h3_mask else -1,

            'CLASSIFICATIONS/EVENT/signal': 1,

        }
        data_list.append(data_dict)

    return data_list


def from_root_to_h5(root_path, output_path, nbj_min=0):
    # root_path: input root file path
    # output_path: output h5 file path
    # nbj_min: 最少要有幾個 b-jet

    f = ROOT.TFile(root_path)
    nevent = f.Get("Delphes").GetEntries()
    print(f'Number of events: {nevent}')

    # Multi-core processing
    print(f'Number of cores: {N_CORES}')
    start = [nevent // N_CORES * i for i in range(N_CORES)]
    end = [nevent // N_CORES * (i+1) for i in range(N_CORES)]
    end[-1] = nevent

    with mp.Pool(processes=N_CORES) as pool:
        results = pool.starmap(select_event, zip(repeat(root_path), repeat(nbj_min), start, end))
    data_list = [data_dict for result_list in results for data_dict in result_list]

    # write to h5 file
    with h5py.File(output_path, 'w') as f_out:
        create_triHiggs_dataset(f_out, nevent, MAX_JETS)
        write_dataset(f_out, data_list)


if __name__ == '__main__':

    root_path = sys.argv[1]
    output_path = sys.argv[2]
    nbj_min = int(sys.argv[3])

    from_root_to_h5(root_path, output_path, nbj_min)
