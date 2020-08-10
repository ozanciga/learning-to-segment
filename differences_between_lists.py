import glob
import re


problematic_annotations = [
101333,
101358,
101364,
101365,
101367,
101373,
101382,
101484,
101489,
101493,
101498,
101499,
99189,
99190,
99191,
99192,
99193,
99205,
99206,
99207,
99208,
100185,
100184,
100179,
100269,
100272,
100273,
100274,
100275,
100277,
100220,
100215,
100212,
100223,
100231,
100222,
100225,
100338,
100339,
101316,
101317,
101368,
101372,
99736,
99757,
99759,
99938,
99939,
99940,
99941,
100186,
100236,
100237,
101511,
101502,
101503,
101504,
101505,
101506,
101507,
101508,
101509,
101510,
99923,
99924,
99925,
99926,
99927,
99928,
99929,
99930,
99931,
99932,
99933,
99934,
99935,
99934,
100166,
100234,
100340,
100216,
99758,
]

previous_list = [99194,
99207,
99191,
99202,
99196,
101495,
101491,
101503,
101487,
101496,
99752,
99758,
99749,
99754,
99747,
99920,
99932,
99915,
99923,
99902,
100166,
100184,
100176,
100174,
100180,
100106,
100120,
100107,
100117,
100109,
100271,
100257,
100270,
100273,
100272,
100333,
100320,
100334,
100335,
100332,
101319,
101318,
101328,
101317,
101330,
101371,
101382,
101378,
101375,
101359,
]


files = glob.glob('/home/ozan/remoteDir/AnnotationSetShawna/*.svs')
out = [int(re.findall('/([0-9]+).svs', s)[0]) for s in files]

problematic_annotations = set(problematic_annotations)

previous_list = set(previous_list)
out = set(out)

list_intersection = sorted(list(out.intersection(previous_list)))

print('intersection count: ', len(list_intersection))
print(list_intersection)

list_diff = sorted(list(out.difference(previous_list)))
print('difference count: ', len(list_diff))
print(list_diff)


list_diff = list(previous_list.intersection(problematic_annotations))

print(sorted(list_diff))

99923
99932
101503
