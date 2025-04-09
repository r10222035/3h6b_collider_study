import sys
import ROOT

import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.append('..')
import utils_HDF5 as utils

delphes_path = '/usr/local/Delphes-3.4.2'
ROOT.gROOT.ProcessLine(f'.include {delphes_path}/')
ROOT.gROOT.ProcessLine(f'.include {delphes_path}/external/')
ROOT.gInterpreter.Declare(f'#include "{delphes_path}/classes/DelphesClasses.h"')
ROOT.gSystem.Load(f'{delphes_path}/install/lib/libDelphes')

MAX_JETS = 15


def DeltaR(eta1, phi1, eta2, phi2):
    dEta = eta1 - eta2
    dPhi = np.abs(phi1 - phi2)

    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)

    dR = np.sqrt(dPhi**2 + dEta**2)
    return dR


def construct_cutflow_table(counts, csv_path=None):
    # construct cutflow table
    cuts = []
    nevents = []
    efficiencies = []
    passing_rates = []
    # construct matching rate table
    n_tot = counts['total']
    n_previous = counts['total']
    for key, value in counts.items():
        cuts.append(key)
        nevents.append(value)
        efficiencies.append(value / n_previous)
        passing_rates.append(value / n_tot)
        if key not in ['3 Higgs', '2 Higgs', '1 Higgs']:
            n_previous = value

    results = {
        'Cuts': cuts,
        'counts': nevents,
        'efficiency': efficiencies,
        'passing rate': passing_rates,
    }
    if csv_path:
        counts_df = pd.DataFrame(results)
        counts_df.to_csv(csv_path, index=False)


def select_event(root_path, nbj_min, start, end, csv_path=None):
    counts = {
        'total': 0,
        '>= 6 jets': 0,
        '>= 4 jets with pT > 40 GeV': 0,
        '>= 4 b-jets': 0,
        '3 Higgs': 0,
        '2 Higgs': 0,
        '1 Higgs': 0,
        '>= 6 b-jets': 0,
    }

    total_event_bjet = np.zeros(MAX_JETS + 5)
    matched_event_bjet = np.zeros(MAX_JETS + 5)
    total_event_Njet = np.zeros(MAX_JETS + 5)
    matched_event_Njet = np.zeros(MAX_JETS + 5)
    f = ROOT.TFile(root_path)
    tree = f.Get("Delphes")

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

        counts['total'] += 1
        # 至少要 6 jet
        if nj < 6:
            continue
        counts['>= 6 jets'] += 1
        # 至少要 4 jet pT > 40 GeV
        if eta_pt40_cut.sum() < 4:
            continue
        counts['>= 4 jets with pT > 40 GeV'] += 1
        nbj = np.array(jet_BTag[eta_pt_cut][:MAX_JETS]).sum()
        # 在前 MAX_JETS jets 中，至少要 nbj_min 個 b-jet
        if nbj < nbj_min:
            continue
        counts['>= 4 b-jets'] += 1
        total_event_bjet[nbj] += 1
        total_event_Njet[nj] += 1

        # 如果 jet 數目超過，則只考慮前 MAX_JETS jets
        # PT = np.array(jet_PT[eta_pt_cut])[:MAX_JETS]
        Eta = np.array(jet_Eta[eta_pt_cut])[:MAX_JETS]
        Phi = np.array(jet_Phi[eta_pt_cut])[:MAX_JETS]
        # Mass = np.array(jet_Mass[eta_pt_cut])[:MAX_JETS]
        # BTag = np.array(jet_BTag[eta_pt_cut])[:MAX_JETS]

        # 找出每個夸克配對的 jet
        # more_than_1_jet = False
        for quark in range(len(quarks_Jet)):
            dR = DeltaR(quarks_Eta[quark], quarks_Phi[quark], Eta, Phi)
            if dR.min() < 0.4:
                quarks_Jet[quark] = np.argmin(dR)


        quark_jet = quarks_Jet.reshape(1, 6)

        h1_mask = utils.get_particle_mask(quark_jet, [0, 1])
        h2_mask = utils.get_particle_mask(quark_jet, [2, 3])
        h3_mask = utils.get_particle_mask(quark_jet, [4, 5])

        nh = [h1_mask, h2_mask, h3_mask].count(True)

        if nh == 3:
            counts['3 Higgs'] += 1
            matched_event_bjet[nbj] += 1
            matched_event_Njet[nj] += 1
        elif nh == 2:
            counts['2 Higgs'] += 1
        elif nh == 1:
            counts['1 Higgs'] += 1

        if nbj >= 6:
            counts['>= 6 b-jets'] += 1

    # save counts to csv
    if csv_path:
        construct_cutflow_table(counts, csv_path)

def compute_cutflow_table(root_path, nbj_min=0, csv_path=None):
    f = ROOT.TFile(root_path)
    nevent = f.Get("Delphes").GetEntries()
    print(f'Number of events: {nevent}')

    start = 0
    end = nevent

    select_event(root_path, nbj_min, start, end, csv_path)


if __name__ == '__main__':
    root_path = sys.argv[1]
    nbj_min = int(sys.argv[2])
    csv_path = sys.argv[3]
    compute_cutflow_table(root_path, nbj_min=nbj_min, csv_path=csv_path)