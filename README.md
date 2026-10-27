data: mmm_example.csv  
source: https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv
  
Date columns: `date_week` and `dayofyear`
Sales column: `y`
Channel media spend columns: `x1` and `x2`
Event columns: `event_1` and `event_2`
Linear trend column: `t`

Shape: (179, 8)

Columns and types:
date_week        str
y            float64
x1           float64
x2           float64
event_1      float64
event_2      float64
dayofyear      int64
t              int64
dtype: object

First 5 rows:
    date_week            y        x1   x2  event_1  event_2  dayofyear  t
0  2018-04-02  3984.662237  0.318580  0.0      0.0      0.0         92  0
1  2018-04-09  3762.871794  0.112388  0.0      0.0      0.0         99  1
2  2018-04-16  4466.967388  0.292400  0.0      0.0      0.0        106  2
3  2018-04-23  3864.219373  0.071399  0.0      0.0      0.0        113  3
4  2018-04-30  4441.625278  0.386745  0.0      0.0      0.0        120  4

Missing values:
date_week    0
y            0
x1           0
x2           0
event_1      0
event_2      0
dayofyear    0
t            0

Numeric summary:
                 y          x1          x2     event_1     event_2   dayofyear           t
count   179.000000  179.000000  179.000000  179.000000  179.000000  179.000000  179.000000
mean   5284.947576    0.309082    0.161850    0.005587    0.005587  182.597765   89.000000
std    1087.276350    0.237638    0.348671    0.074744    0.074744  100.370019   51.816986
min    3192.879593    0.000929    0.000000    0.000000    0.000000    4.000000    0.000000
25%    4484.221415    0.154219    0.000000    0.000000    0.000000  103.000000   44.500000
50%    5163.385726    0.274762    0.000000    0.000000    0.000000  181.000000   89.000000
75%    5941.275944    0.397176    0.000000    0.000000    0.000000  262.500000  133.500000
max    8312.407544    0.996658    0.994374    1.000000    1.000000  365.000000  178.000000