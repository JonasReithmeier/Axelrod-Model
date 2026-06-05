import numpy as np

L=np.array([30,40])
N = L**2
print(N)
int_upper_lim = 0.1
data_pts = 20

max_q = max(int_upper_lim*N)

q_list = []

q_per_point = round(max_q / data_pts, 0)
for i in range(data_pts):
    q_list.append(int(q_per_point * i))

print(q_list)

'''
as_experiment:
  master_seed: 42
  M_realizations: 10
  max_mcs: 1000000
  transient_mcs: 5000 # can be cut down to 5k ??? look at mobility plots
  sweep:
    L: [30, 40]
    F: [10]

    q:
      [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48,
        50,
        52,
        54,
        56,
        58,
        60,
        62,
        64,
        66,
        68,
        70,
        72,
        74,
        76,
        78,
        80,
        82,
        84,
        86,
        88,
        90,
        92,
        94,
        96,
        98,
        100,
        102,
        104,
        106,
        108,
        110,
        112,
        114,
        116,
        118,
        120,
        122,
        124,
        126,
        128,
        130,
        132,
        134,
        136,
        138,
        140,
        142,
        144,
        146,
        148,
        150,
        152,
        154,
        156,
        158,
        160,
        162,
        164,
        166,
        168,
        170,
        172,
        174,
        176,
        178,
        180,
        182,
        184,
        186,
        188,
        190,
        192,
        194,
        196,
        198,
      ]

    # h = Fraction of empty sites
    h: [0.05] #   , 0.3, 0.5, 0.7

    # T = Tolerance
    T: [1.0] #    0.0, 0.3, 0.5, 0.7,

'''