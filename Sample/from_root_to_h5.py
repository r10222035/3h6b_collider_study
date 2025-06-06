#!/usr/bin/env python
# coding: utf-8
# generate tri-Higgs HDF5 data for SPANet
# jets are required PT > 20 GeV
# with correct jet assignment
# python from_root_to_h5.py <root file path> <output file name> <minimum b-jet>

import sys
import h5py
import ROOT

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from itertools import repeat

sys.path.append('..')
import utils_HDF5 as utils

delphes_path = '/usr/local/Delphes-3.4.2/'
ROOT.gROOT.ProcessLine(f'.include {delphes_path}')
ROOT.gROOT.ProcessLine(f'.include {delphes_path}external/')
ROOT.gInterpreter.Declare(f'#include "{delphes_path}classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare(f'#include "{delphes_path}external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare(f'#include "{delphes_path}external/ExRootAnalysis/ExRootConfReader.h"')
ROOT.gInterpreter.Declare(f'#include "{delphes_path}external/ExRootAnalysis/ExRootTask.h"')
ROOT.gSystem.Load(f'{delphes_path}install/lib/libDelphes')

MAX_JETS = 15
N_CORES = 64


def DeltaR(eta1, phi1, eta2, phi2):
    dEta = eta1 - eta2
    dPhi = np.abs(phi1 - phi2)
    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)

    dR = np.sqrt(dPhi**2 + dEta**2)
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
        quarks_Jet = np.array([-1, -1, -1, -1, -1, -1])

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

        # 找出 6 個 final b quark
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
        # |eta| < 2.5 & PT > 20 GeV
        eta_pt_cut = np.array((np.abs(jet_Eta) < 2.5) & (jet_PT > 20))
        # |eta| < 2.5 & PT > 40 GeV
        eta_pt40_cut = np.array((np.abs(jet_Eta) < 2.5) & (jet_PT > 40))

        nj = eta_pt_cut.sum()

        # 至少要 6 jet
        if nj < 6:
            continue

        # 至少要 4 jet pT > 40 GeV
        if eta_pt40_cut.sum() < 4:
            continue
        
        nbj = np.array(jet_BTag[eta_pt_cut][:MAX_JETS]).sum()
        # 在前 MAX_JETS jets 中，至少要 nbj_min 個 b-jet
        if nbj < nbj_min:
            continue

        # 如果 jet 數目超過，則只考慮前 MAX_JETS jets
        PT = np.array(jet_PT[eta_pt_cut])[:MAX_JETS]
        Eta = np.array(jet_Eta[eta_pt_cut])[:MAX_JETS]
        Phi = np.array(jet_Phi[eta_pt_cut])[:MAX_JETS]
        Mass = np.array(jet_Mass[eta_pt_cut])[:MAX_JETS]
        BTag = np.array(jet_BTag[eta_pt_cut])[:MAX_JETS]

        # 找出每個夸克配對的 jet
        for quark in range(len(quarks_Jet)):
            dR = DeltaR(quarks_Eta[quark], quarks_Phi[quark], Eta, Phi)
            if dR.min() < 0.4:
                quarks_Jet[quark] = np.argmin(dR)

        quark_jet = quarks_Jet.reshape(1, 6)

        h1_mask = utils.get_particle_mask(quark_jet, [0, 1])
        h2_mask = utils.get_particle_mask(quark_jet, [2, 3])
        h3_mask = utils.get_particle_mask(quark_jet, [4, 5])

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