#!/usr/bin/env python
# coding: utf-8
# generate tri-Higgs HDF5 data for SPANet
# jets are required PT > 25 GeV
# with correct jet assignment
# python from_root_to_h5.py <root file path> <output file name> <minimum b-jet>

import os
import sys
import h5py
import math
import uproot

import numpy as np

from tqdm import tqdm


class BranchGenParticles:
    def __init__(self, file, start, end):
        print('Initialize GenParticles')
        self.file = file
        self.length = len(file['Particle.Status'].array(entry_start=start, entry_stop=end))
        print('Initialize GenParticles: Status')
        self.Status = file['Particle.Status'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: PID')
        self.PID = file['Particle.PID'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: M1')
        self.M1 = file['Particle.M1'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: M2')
        self.M2 = file['Particle.M2'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: D1')
        self.D1 = file['Particle.D1'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: D2')
        self.D2  = file['Particle.D2'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: PT')
        self.PT = file['Particle.PT'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Eta')
        self.Eta =  file['Particle.Eta'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Phi')
        self.Phi = file['Particle.Phi'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Mass')
        self.Mass = file['Particle.Mass'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Charge')
        self.Charge = file['Particle.Charge'].array(entry_start=start, entry_stop=end)
        self.Labels = ['Status', 'PID', 'M1', 'M2', 'D1', 'D2', 'PT', 'Eta', 'Phi', 'Mass', 'Charge']

    def length_At(self, i):
        return len(self.Status[i])

    def Status_At(self, i):
        return self.Status[i]

    def PID_At(self, i):
        return self.PID[i]

    def M1_At(self, i):
        return self.M1[i]

    def M2_At(self, i):
        return self.M2[i]

    def D1_At(self, i):
        return self.D1[i]

    def D2_At(self, i):
        return self.D2[i]

    def PT_At(self, i):
        return self.PT[i]

    def Eta_At(self, i):
        return self.Eta[i]

    def Phi_At(self, i):
        return self.Phi[i]

    def Mass_At(self, i):
        return self.Mass[i]

    def Charge_At(self, i):
        return self.Charge[i]


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


def write_dataset(file, index, data):
    # data: dictionary
    for key, value in data.items():
        file[key][index] = value


def get_dataset_keys(f):
    # 取得所有 Dataset 的名稱
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def resize_h5(file_path, nevent):
    with h5py.File(file_path, 'r+') as f:
        datasets = get_dataset_keys(f)
        for dataset in datasets:
            shape = list(f[dataset].shape)
            shape[0] = nevent
            f[dataset].resize(shape)
    print(f'{file_path} resize to {nevent}')


def main(root_path, output_path, nbj_min=0, nevent_max=100000):
    # root_path: input root file path
    # output_path: output h5 file path
    # nbj_min: 最少要有幾個 b-jet
    # nevent_max: 每個 h5 file 最多有幾個 event，記憶體不夠時可以調小

    MAX_JETS = 15

    root_file = uproot.open(root_path)["Delphes;1"]
    num_entries = root_file.num_entries

    for i in range(math.ceil(num_entries/nevent_max)):

        start = i * nevent_max
        end = min(start + nevent_max, num_entries)

        GenParticle = BranchGenParticles(root_file, start, end)

        jet_PT = root_file['Jet.PT'].array(entry_start=start, entry_stop=end)
        jet_Eta = root_file['Jet.Eta'].array(entry_start=start, entry_stop=end)
        jet_Phi = root_file['Jet.Phi'].array(entry_start=start, entry_stop=end)
        jet_Mass = root_file['Jet.Mass'].array(entry_start=start, entry_stop=end)
        jet_BTag = root_file['Jet.BTag'].array(entry_start=start, entry_stop=end)

        nevent = len(jet_PT)
        event_index = 0

        root, _ = os.path.splitext(output_path)
        event_file_path = f'{root}-{i:02}.h5'

        with h5py.File(event_file_path, 'w') as f_out:

            create_triHiggs_dataset(f_out, nevent, MAX_JETS)

            for event in tqdm(range(nevent)):

                # 夸克資料
                # b夸克 衰變前的編號
                quarks_id = []
                quarks_Eta = []
                quarks_Phi = []
                quarks_Jet = [-1, -1, -1, -1, -1, -1]

                PID = GenParticle.PID_At(event)
                D1 = GenParticle.D1_At(event)
                D2 = GenParticle.D2_At(event)

                # 找出 3 個 final Higgs
                final_h_index = []
                for j in np.where(PID == 25)[0]:
                    h = j
                    d1 = D1[h]
                    while abs(PID[d1]) == 25:
                        h = d1
                        d1 = D1[h]
                    final_h_index.append(h)

                final_h_index = list(set(final_h_index))

                # 找出 6個 final b quark
                for h in final_h_index:
                    # h > b b~
                    b1 = D1[h]
                    b2 = D2[h]

                    # 找出 b 衰變前的編號
                    d1 = D1[b1]
                    while abs(PID[d1]) == 5:
                        b1 = d1
                        d1 = D1[b1]

                    # 找出 b~ 衰變前的編號
                    d2 = D1[b2]
                    while abs(PID[d2]) == 5:
                        b2 = d2
                        d2 = D1[b2]

                    quarks_id.extend([b1, b2])

                quarks_Eta.extend(GenParticle.Eta_At(event)[quarks_id])
                quarks_Phi.extend(GenParticle.Phi_At(event)[quarks_id])

                # Jet 資料
                # |eta| < 2.5 & PT > 25 GeV
                eta_pt_cut = np.array((np.abs(jet_Eta[event]) < 2.5) & (jet_PT[event] > 25))

                nj = eta_pt_cut.sum()

                # 至少要 6 jet
                if nj < 6:
                    continue

                nbj = np.array(jet_BTag[event][eta_pt_cut][:MAX_JETS]).sum()
                # 在前 MAX_JETS jets 中，至少要 nbj_min 個 b-jet
                if nbj < nbj_min:
                    continue

                PT = np.array(jet_PT[event][eta_pt_cut])
                Eta = np.array(jet_Eta[event][eta_pt_cut])
                Phi = np.array(jet_Phi[event][eta_pt_cut])
                Mass = np.array(jet_Mass[event][eta_pt_cut])
                BTag = np.array(jet_BTag[event][eta_pt_cut])

                # 找出每個夸克配對的 jet
                more_than_1_jet = False
                for quark in range(len(quarks_Jet)):
                    for i in range(min(nj, MAX_JETS)):
                        dR = DeltaR(Eta[i], Phi[i], quarks_Eta[quark], quarks_Phi[quark])
                        if dR < 0.4 and quarks_Jet[quark] == -1:
                            quarks_Jet[quark] = i
                        elif dR < 0.4:
                            more_than_1_jet = True

                if more_than_1_jet: continue

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

                write_dataset(f_out, event_index, data_dict)
                event_index += 1

        resize_h5(event_file_path, event_index)


if __name__ == '__main__':

    root_path = sys.argv[1]
    output_path = sys.argv[2]
    nbj_min = int(sys.argv[3])

    main(root_path, output_path, nbj_min)