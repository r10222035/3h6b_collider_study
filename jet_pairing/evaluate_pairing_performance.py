import os
import sys
import h5py

import numpy as np
import pandas as pd

from tqdm import tqdm

def get_Higgs_result(total_event, total_Higgs, correct_event, correct_Higgs, nh, nj):

    start_nj, end_nj = nj

    label = ['all' if nh == 'all' else f'{nh}h' for nj in range(start_nj, end_nj+2)]

    if nh == 'all':
        nh = slice(1, None)
        
    event_type = [f'Nj={nj}' for nj in range(start_nj, end_nj)]
    event_type.append(f'Nj>={end_nj}')
    event_type.append('Total')

    event_fraction = [total_event[nh, nj].sum() / total_event.sum() for nj in range(start_nj, end_nj)]
    event_fraction.append(total_event[nh, end_nj:].sum() / total_event.sum())
    event_fraction.append(total_event[nh].sum() / total_event.sum())

    event_efficiency = [correct_event[nh, nj].sum() / total_event[nh, nj].sum() for nj in range(start_nj, end_nj)]
    event_efficiency.append(correct_event[nh, end_nj:].sum() / total_event[nh, end_nj:].sum())
    event_efficiency.append(correct_event[nh].sum() / total_event[nh].sum())

    higgs_efficiency = [correct_Higgs[nh, nj].sum() / total_Higgs[nh, nj].sum() for nj in range(start_nj, end_nj)]
    higgs_efficiency.append(correct_Higgs[nh, end_nj:].sum() / total_Higgs[nh, end_nj:].sum())
    higgs_efficiency.append(correct_Higgs[nh].sum() / total_Higgs[nh].sum())
        
    result = {'Label':label,
              'Event type': event_type,
              'Event Fraction': event_fraction,
              'Event Efficiency': event_efficiency,
              'Higgs Efficiency': higgs_efficiency,
             }

    df = pd.DataFrame(result)

    return df


def compare_jet_list_triHiggs_optimized(pair1, pair2, nh_max=3):
    # 將pair1和pair2分別轉換為三個Higgs的集合
    h_true_sets = [{pair1[i], pair1[i+1]} for i in range(0, 6, 2)]
    h_test_sets = [{pair2[i], pair2[i+1]} for i in range(0, 6, 2)]
    
    # 計算匹配的Higgs數量
    nh = sum(1 for h_true in h_true_sets if h_true in h_test_sets)
    
    # 判斷是否所有Higgs都匹配
    same = nh == nh_max
    return same, nh


def get_particle_mask(quark_jet, particle_quarks):
    # quark_jet: 每個夸克對應的 jet 編號，shape 為 (n_event, 6)
    # particle_quarks: 粒子對應的夸克編號，shape 為 (n_quarks,)

    # 檢查是否每個夸克都有對應的 jet
    mask1 = np.all(quark_jet[:, particle_quarks] != -1, axis=1)

    # 對每一個事件，檢查每個夸克對應的 jet 都不重複
    count = np.array([[np.sum(event == event[i]) for i in particle_quarks] for event in quark_jet])
    mask2 = np.all(count == 1, axis=1)

    return mask1 & mask2


def get_Higgs_correct_fraction(events, nh, nj, jet_type='Nj'):
    # events: number of events in different categories (nh, nj, n_correct_h)
    start_nj, end_nj = nj

    label = [f'{nh}h' for _ in range(start_nj, end_nj+2)]

    total_event = events[nh].sum(axis=1)

    correct_3h_event = events[nh, :, 3]
    correct_2h_event = events[nh, :, 2]
    correct_1h_event = events[nh, :, 1]
    correct_0h_event = events[nh, :, 0]

    correct_Higgs = events[nh, :, 3] * 3 + events[nh, :, 2] * 2 + events[nh, :, 1]
        
    event_type = [f'{jet_type}={nj}' for nj in range(start_nj, end_nj)]
    event_type.append(f'{jet_type}>={end_nj}')
    event_type.append('Total')

    event_fraction = [total_event[nj] / total_event.sum() for nj in range(start_nj, end_nj)]
    event_fraction.append(total_event[end_nj:].sum() / total_event.sum())
    event_fraction.append(total_event.sum() / total_event.sum())


    eff_3h = [correct_3h_event[nj] / total_event[nj] for nj in range(start_nj, end_nj)]
    eff_3h.append(correct_3h_event[end_nj:].sum() / total_event[end_nj:].sum())
    eff_3h.append(correct_3h_event.sum() / total_event.sum())

    eff_2h = [correct_2h_event[nj] / total_event[nj] for nj in range(start_nj, end_nj)]
    eff_2h.append(correct_2h_event[end_nj:].sum() / total_event[end_nj:].sum())
    eff_2h.append(correct_2h_event.sum() / total_event.sum())

    eff_1h = [correct_1h_event[nj] / total_event[nj] for nj in range(start_nj, end_nj)]
    eff_1h.append(correct_1h_event[end_nj:].sum() / total_event[end_nj:].sum())
    eff_1h.append(correct_1h_event.sum() / total_event.sum())

    eff_0h = [correct_0h_event[nj] / total_event[nj] for nj in range(start_nj, end_nj)]
    eff_0h.append(correct_0h_event[end_nj:].sum() / total_event[end_nj:].sum())
    eff_0h.append(correct_0h_event.sum() / total_event.sum())

    eff_Higgs = [correct_Higgs[nj] / (total_event[nj] * nh) for nj in range(start_nj, end_nj)]
    eff_Higgs.append(correct_Higgs[end_nj:].sum() / (total_event[end_nj:].sum() * nh))
    eff_Higgs.append(correct_Higgs.sum() / (total_event.sum() * nh))
        
    result = {'Label':label,
              'Event type': event_type,
              'Event Fraction': event_fraction,
              '3h': eff_3h,
              '2h': eff_2h,
              '1h': eff_1h,
              '0h': eff_0h,
              'Higgs': eff_Higgs
             }

    df = pd.DataFrame(result)

    return df


# 載入正確配對與測試配對的資料，並計算配對的效率
def compute_pairing_efficiency(true_file, test_file, save_path=None):
    MAX_JETS = 15

    with h5py.File(true_file, 'r') as f_true, h5py.File(test_file, 'r') as f_test:
        
        # events: 總共有多少該類事件 (nh, nj, n_correct_h)
        events = np.zeros((4, MAX_JETS + 1, 4))

        # nevent = f_true['INPUTS/Source/pt'].shape[0]
        nevent = min(100000, f_true['INPUTS/Source/pt'].shape[0])

        for event in tqdm(range(nevent)):

            nj = f_true['INPUTS/Source/MASK'][event].sum()

            h1_b1 = f_true['TARGETS/h1/b1'][event]
            h1_b2 = f_true['TARGETS/h1/b2'][event]
            h2_b1 = f_true['TARGETS/h2/b1'][event]
            h2_b2 = f_true['TARGETS/h2/b2'][event]
            h3_b1 = f_true['TARGETS/h3/b1'][event]
            h3_b2 = f_true['TARGETS/h3/b2'][event]

            quark_jet = np.array([h1_b1, h1_b2, h2_b1, h2_b2, h3_b1, h3_b2]).reshape(1, 6)

            h1_mask = get_particle_mask(quark_jet, [0, 1])
            h2_mask = get_particle_mask(quark_jet, [2, 3])
            h3_mask = get_particle_mask(quark_jet, [4, 5])

            event_h = [h1_mask, h2_mask, h3_mask].count(True)

            true_pair = [h1_b1,h1_b2, h2_b1,h2_b2, h3_b1,h3_b2]

            try: 
                h1_b1 = f_test['TARGETS/h1/b1'][event]
                h1_b2 = f_test['TARGETS/h1/b2'][event]
                h2_b1 = f_test['TARGETS/h2/b1'][event]
                h2_b2 = f_test['TARGETS/h2/b2'][event]
                h3_b1 = f_test['TARGETS/h3/b1'][event]
                h3_b2 = f_test['TARGETS/h3/b2'][event]
            except KeyError:
                h1_b1 = f_test['SpecialKey.Targets/h1/b1'][event]
                h1_b2 = f_test['SpecialKey.Targets/h1/b2'][event]
                h2_b1 = f_test['SpecialKey.Targets/h2/b1'][event]
                h2_b2 = f_test['SpecialKey.Targets/h2/b2'][event]
                h3_b1 = f_test['SpecialKey.Targets/h3/b1'][event]
                h3_b2 = f_test['SpecialKey.Targets/h3/b2'][event]
            pair = [h1_b1,h1_b2, h2_b1,h2_b2, h3_b1,h3_b2]


            if event_h == 3:
                _, nh = compare_jet_list_triHiggs_optimized(true_pair, pair, nh_max=3)
                events[3, nj, nh] += 1
            elif event_h == 2:
                _, nh = compare_jet_list_triHiggs_optimized(true_pair, pair, nh_max=2)
                events[2, nj, nh] += 1
            elif event_h == 1:
                _, nh = compare_jet_list_triHiggs_optimized(true_pair, pair, nh_max=1)
                events[1, nj, nh] += 1
            elif event_h == 0:
                events[0, nj, 0] += 1
        
        
        df_3h = get_Higgs_correct_fraction(events, nh=3, nj=(6, 8), jet_type='Nj')  
        # print('3 Higgs Events:')
        # print(df_3h)
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_3h.to_csv(save_path, index=False)

        df_style = df_3h.style.format({
            'Event Fraction': '{:.3f}',
            '3h': '{:.3f}',
            '2h': '{:.3f}',
            '1h': '{:.3f}',
            '0h': '{:.3f}',
            'Higgs': '{:.3f}'
        })
        # print(df_style.to_latex(column_format='c|cccc|c'))


if __name__ == '__main__':

    true_path = sys.argv[1]
    predict_path = sys.argv[2]
    csv_path = sys.argv[3]

    compute_pairing_efficiency(true_path, predict_path, csv_path)
