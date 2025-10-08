# MIxS Environmental Triad Analysis Report

Generated: 2025-10-08 13:57:42


## Cross-Dataset Comparison


### env_broad_scale

| Metric                 | GOLD     | NCBI      | NMDC     |
|:-----------------------|:---------|:----------|:---------|
| Total Records          | 11,083   | 452,876   | 8,434    |
| Unique Terms           | 42       | 622       | 21       |
| Terms < 0.1%           | 11       | 482       | 6        |
| Terms for 50% coverage | 3 (7.1%) | 21 (3.4%) | 2 (9.5%) |



### env_local_scale

| Metric                 | GOLD      | NCBI      | NMDC     |
|:-----------------------|:----------|:----------|:---------|
| Total Records          | 11,083    | 452,876   | 8,434    |
| Unique Terms           | 150       | 891       | 54       |
| Terms < 0.1%           | 59        | 696       | 13       |
| Terms for 50% coverage | 14 (9.3%) | 36 (4.0%) | 4 (7.4%) |



### env_medium

| Metric                 | GOLD     | NCBI      | NMDC     |
|:-----------------------|:---------|:----------|:---------|
| Total Records          | 11,083   | 452,876   | 8,434    |
| Unique Terms           | 81       | 618       | 32       |
| Terms < 0.1%           | 30       | 492       | 5        |
| Terms for 50% coverage | 5 (6.2%) | 13 (2.1%) | 2 (6.2%) |



## Detailed Frequency Analysis


### GOLD


#### env_broad_scale

- Total records: 11,083

- Unique terms: 42

- Terms after filter: 31

- **Imbalance Severity:** 🟡 HIGH

- Terms for 50% coverage: 3 (7.1%)

- Terms for 80% coverage: 8 (19.1%)

- Terms for 95% coverage: 17 (40.5%)


**Top 20 Most Frequent Terms:**


| term          | label                           |   count |   percentage |
|:--------------|:--------------------------------|--------:|-------------:|
| ENVO:00000446 | terrestrial biome               |    3844 |       34.684 |
| ENVO:01000252 | freshwater lake biome           |    1071 |        9.663 |
| ENVO:01000174 | forest biome                    |     961 |        8.671 |
| ENVO:00000447 | marine biome                    |     907 |        8.184 |
| ENVO:01000253 | freshwater river biome          |     883 |        7.967 |
| ENVO:00002030 | aquatic biome                   |     648 |        5.847 |
| ENVO:00000873 | freshwater biome                |     467 |        4.214 |
| ENVO:01000177 | grassland biome                 |     346 |        3.122 |
| ENVO:01000219 | anthropogenic terrestrial biome |     215 |        1.94  |
| ENVO:01000023 | marine pelagic biome            |     189 |        1.705 |
| ENVO:01000024 | marine benthic biome            |     177 |        1.597 |
| ENVO:01001001 | plant-associated environment    |     175 |        1.579 |
| ENVO:01000339 | polar biome                     |     175 |        1.579 |
| ENVO:01000020 | estuarine biome                 |     161 |        1.453 |
| ENVO:01000249 | urban biome                     |     140 |        1.263 |
| ENVO:01000179 | desert biome                    |     133 |        1.2   |
| ENVO:01000181 | mangrove biome                  |      87 |        0.785 |
| ENVO:01000035 | oceanic epipelagic zone biome   |      86 |        0.776 |
| ENVO:01000127 | marine cold seep biome          |      84 |        0.758 |
| ENVO:01001837 | subalpine biome                 |      51 |        0.46  |



⚠️ **Note:** 26.2% of terms have <0.1% frequency

- Many rare classes may need special handling


#### env_local_scale

- Total records: 11,083

- Unique terms: 150

- Terms after filter: 91

- **Imbalance Severity:** 🟡 HIGH

- Terms for 50% coverage: 14 (9.3%)

- Terms for 80% coverage: 37 (24.7%)

- Terms for 95% coverage: 71 (47.3%)


**Top 20 Most Frequent Terms:**


| term          | label                                      |   count |   percentage |
|:--------------|:-------------------------------------------|--------:|-------------:|
| ENVO:00000022 | river                                      |     857 |        7.733 |
| ENVO:00000021 | freshwater lake                            |     731 |        6.596 |
| ENVO:00000114 | agricultural field                         |     695 |        6.271 |
| ENVO:00000051 | hot spring                                 |     449 |        4.051 |
| ENVO:00000044 | peatland                                   |     440 |        3.97  |
| ENVO:01000843 | area of evergreen forest                   |     381 |        3.438 |
| ENVO:00000234 | bayou                                      |     327 |        2.95  |
| ENVO:01001275 | solid layer                                |     311 |        2.806 |
| ENVO:00005801 | rhizosphere                                |     302 |        2.725 |
| ENVO:01000816 | area of deciduous forest                   |     281 |        2.535 |
| ENVO:01001785 | land                                       |     266 |        2.4   |
| ENVO:00000292 | watershed                                  |     246 |        2.22  |
| ENVO:00002169 | coal mine                                  |     233 |        2.102 |
| ENVO:01000159 | obsolete organic feature                   |     212 |        1.913 |
| ENVO:01000888 | area of gramanoid or herbaceous vegetation |     207 |        1.868 |
| ENVO:00001999 | marine water body                          |     194 |        1.75  |
| ENVO:00000043 | wetland area                               |     186 |        1.678 |
| ENVO:00000015 | ocean                                      |     178 |        1.606 |
| ENVO:01000349 | root matter                                |     175 |        1.579 |
| ENVO:00000111 | forested area                              |     174 |        1.57  |



⚠️ **Note:** 39.3% of terms have <0.1% frequency

- Many rare classes may need special handling


#### env_medium

- Total records: 11,083

- Unique terms: 81

- Terms after filter: 51

- **Imbalance Severity:** 🟡 HIGH

- Terms for 50% coverage: 5 (6.2%)

- Terms for 80% coverage: 15 (18.5%)

- Terms for 95% coverage: 32 (39.5%)


**Top 20 Most Frequent Terms:**


| term          | label                  |   count |   percentage |
|:--------------|:-----------------------|--------:|-------------:|
| ENVO:00001998 | soil                   |    2221 |       20.04  |
| ENVO:00002007 | sediment               |    1099 |        9.916 |
| ENVO:04000007 | lake water             |     816 |        7.363 |
| ENVO:00002149 | sea water              |     813 |        7.336 |
| ENVO:00002261 | forest soil            |     686 |        6.19  |
| ENVO:00005774 | peat soil              |     582 |        5.251 |
| ENVO:01000599 | river water            |     460 |        4.151 |
| ENVO:00002011 | fresh water            |     398 |        3.591 |
| ENVO:02000091 | coal                   |     359 |        3.239 |
| ENVO:00002259 | agricultural soil      |     351 |        3.167 |
| ENVO:00005802 | bulk soil              |     317 |        2.86  |
| ENVO:03000033 | marine sediment        |     278 |        2.508 |
| ENVO:00000546 | lake sediment          |     236 |        2.129 |
| ENVO:00005781 | heat stressed soil     |     226 |        2.039 |
| ENVO:00005801 | rhizosphere            |     185 |        1.669 |
| ENVO:00002150 | coastal sea water      |     174 |        1.57  |
| ENVO:01000157 | microbial mat material |     159 |        1.435 |
| ENVO:01000628 | plant litter           |     148 |        1.335 |
| ENVO:02000059 | surface soil           |     137 |        1.236 |
| ENVO:00005750 | grassland soil         |     136 |        1.227 |



⚠️ **Note:** 37.0% of terms have <0.1% frequency

- Many rare classes may need special handling


### NCBI


#### env_broad_scale

- Total records: 452,876

- Unique terms: 622

- Terms after filter: 140

- **Imbalance Severity:** 🔴 SEVERE

- Terms for 50% coverage: 21 (3.4%)

- Terms for 80% coverage: 76 (12.2%)

- Terms for 95% coverage: 201 (32.3%)


**Top 20 Most Frequent Terms:**


| term          | label                     |   count |   percentage |
|:--------------|:--------------------------|--------:|-------------:|
| ENVO:00000428 | biome                     |   44592 |        9.846 |
| ENVO:00001998 | soil                      |   19297 |        4.261 |
| ENVO:00000446 | terrestrial biome         |   17667 |        3.901 |
| ENVO:00000447 | marine biome              |   16334 |        3.607 |
| ENVO:00002030 | aquatic biome             |   12935 |        2.856 |
| ENVO:00000114 | agricultural field        |   12300 |        2.716 |
| ENVO:00000890 | small river biome         |   11044 |        2.439 |
| ENVO:00000029 | watercourse               |    9807 |        2.165 |
| ENVO:01000020 | estuarine biome           |    9638 |        2.128 |
| ENVO:01000174 | forest biome              |    7915 |        1.748 |
| ENVO:00002046 | activated sludge          |    7062 |        1.559 |
| ENVO:00000015 | ocean                     |    7019 |        1.55  |
| ENVO:00000150 | coral reef                |    6685 |        1.476 |
| ENVO:00000111 | forested area             |    5905 |        1.304 |
| ENVO:01000198 | mixed forest biome        |    5790 |        1.278 |
| ENVO:00000051 | hot spring                |    5536 |        1.222 |
| ENVO:00000873 | freshwater biome          |    5531 |        1.221 |
| ENVO:00000020 | lake                      |    5480 |        1.21  |
| ENVO:00000241 | tidal mudflat             |    5463 |        1.206 |
| ENVO:01000313 | anthropogenic environment |    5389 |        1.19  |



⚠️ **DATA QUALITY WARNING:** 77.5% of terms have <0.1% frequency

- Consider: parent class grouping, increasing min_pct threshold, or data augmentation


#### env_local_scale

- Total records: 452,876

- Unique terms: 891

- Terms after filter: 195

- **Imbalance Severity:** 🔴 SEVERE

- Terms for 50% coverage: 36 (4.0%)

- Terms for 80% coverage: 134 (15.0%)

- Terms for 95% coverage: 313 (35.1%)


**Top 20 Most Frequent Terms:**


| term          | label                   |   count |   percentage |
|:--------------|:------------------------|--------:|-------------:|
| ENVO:00001998 | soil                    |   16903 |        3.732 |
| ENVO:00000316 | intertidal zone         |   13807 |        3.049 |
| ENVO:00000114 | agricultural field      |   13380 |        2.954 |
| ENVO:00000111 | forested area           |   13061 |        2.884 |
| ENVO:00000023 | stream                  |   10345 |        2.284 |
| ENVO:00000486 | shoreline               |    9759 |        2.155 |
| ENVO:00000029 | watercourse             |    8904 |        1.966 |
| ENVO:00000077 | agricultural ecosystem  |    7611 |        1.681 |
| ENVO:00005801 | rhizosphere             |    7607 |        1.68  |
| ENVO:00002046 | activated sludge        |    7420 |        1.638 |
| ENVO:00002259 | agricultural soil       |    6656 |        1.47  |
| ENVO:00002001 | waste water             |    6408 |        1.415 |
| ENVO:00000020 | lake                    |    6321 |        1.396 |
| ENVO:01000999 | rhizosphere environment |    6313 |        1.394 |
| ENVO:01001405 | laboratory environment  |    6279 |        1.386 |
| ENVO:00000296 | rice field              |    6048 |        1.335 |
| ENVO:01000352 | field                   |    5791 |        1.279 |
| ENVO:00000022 | river                   |    5730 |        1.265 |
| ENVO:00002150 | coastal sea water       |    5529 |        1.221 |
| ENVO:00000051 | hot spring              |    4597 |        1.015 |



⚠️ **DATA QUALITY WARNING:** 78.1% of terms have <0.1% frequency

- Consider: parent class grouping, increasing min_pct threshold, or data augmentation


#### env_medium

- Total records: 452,876

- Unique terms: 618

- Terms after filter: 126

- **Imbalance Severity:** 🔴 SEVERE

- Terms for 50% coverage: 13 (2.1%)

- Terms for 80% coverage: 51 (8.2%)

- Terms for 95% coverage: 157 (25.4%)


**Top 20 Most Frequent Terms:**


| term          | label                  |   count |   percentage |
|:--------------|:-----------------------|--------:|-------------:|
| ENVO:00001998 | soil                   |   78653 |       17.367 |
| ENVO:00002006 | liquid water           |   23188 |        5.12  |
| ENVO:00002149 | sea water              |   21928 |        4.842 |
| ENVO:00002003 | fecal material         |   16088 |        3.552 |
| ENVO:00002007 | sediment               |   15755 |        3.479 |
| ENVO:00002259 | agricultural soil      |   11997 |        2.649 |
| ENVO:00002261 | forest soil            |    9696 |        2.141 |
| ENVO:00002011 | fresh water            |    9427 |        2.082 |
| ENVO:01000301 | estuarine water        |    9206 |        2.033 |
| ENVO:00002001 | waste water            |    8648 |        1.91  |
| ENVO:00000029 | watercourse            |    8303 |        1.833 |
| ENVO:00002150 | coastal sea water      |    7845 |        1.732 |
| ENVO:00005801 | rhizosphere            |    7082 |        1.564 |
| ENVO:00000111 | forested area          |    7048 |        1.556 |
| ENVO:00005802 | bulk soil              |    7028 |        1.552 |
| ENVO:04000007 | lake water             |    6875 |        1.518 |
| ENVO:00000077 | agricultural ecosystem |    6088 |        1.344 |
| ENVO:03501217 | tissue paper           |    5765 |        1.273 |
| ENVO:00010483 | environmental material |    5440 |        1.201 |
| ENVO:00000241 | tidal mudflat          |    5435 |        1.2   |



⚠️ **DATA QUALITY WARNING:** 79.6% of terms have <0.1% frequency

- Consider: parent class grouping, increasing min_pct threshold, or data augmentation


### NMDC


#### env_broad_scale

- Total records: 8,434

- Unique terms: 21

- Terms after filter: 15

- **Imbalance Severity:** 🟡 HIGH

- Terms for 50% coverage: 2 (9.5%)

- Terms for 80% coverage: 4 (19.1%)

- Terms for 95% coverage: 8 (38.1%)


**Top 20 Most Frequent Terms:**


| term          | label                           |   count |   percentage |
|:--------------|:--------------------------------|--------:|-------------:|
| ENVO:00000446 | terrestrial biome               |    4035 |       47.842 |
| ENVO:01000221 | temperate woodland biome        |    1820 |       21.579 |
| ENVO:01000253 | freshwater river biome          |     750 |        8.893 |
| ENVO:01000245 | cropland biome                  |     712 |        8.442 |
| ENVO:01000036 | oceanic mesopelagic zone biome  |     290 |        3.438 |
| ENVO:01000252 | freshwater lake biome           |     153 |        1.814 |
| ENVO:01001001 | plant-associated environment    |     140 |        1.66  |
| ENVO:01000249 | urban biome                     |     125 |        1.482 |
| ENVO:01000177 | grassland biome                 |     122 |        1.447 |
| ENVO:03605008 | freshwater stream biome         |      92 |        1.091 |
| ENVO:01001442 | agriculture                     |      56 |        0.664 |
| ENVO:01000219 | anthropogenic terrestrial biome |      48 |        0.569 |
| ENVO:01000174 | forest biome                    |      42 |        0.498 |
| ENVO:01001837 | subalpine biome                 |      18 |        0.213 |
| ENVO:01000196 | coniferous forest biome         |      11 |        0.13  |



⚠️ **Note:** 28.6% of terms have <0.1% frequency

- Many rare classes may need special handling


#### env_local_scale

- Total records: 8,434

- Unique terms: 54

- Terms after filter: 41

- **Imbalance Severity:** 🟡 HIGH

- Terms for 50% coverage: 4 (7.4%)

- Terms for 80% coverage: 12 (22.2%)

- Terms for 95% coverage: 25 (46.3%)


**Top 20 Most Frequent Terms:**


| term          | label                                                   |   count |   percentage |
|:--------------|:--------------------------------------------------------|--------:|-------------:|
| ENVO:00000114 | agricultural field                                      |    1824 |       21.627 |
| ENVO:01000843 | area of evergreen forest                                |    1041 |       12.343 |
| ENVO:01000816 | area of deciduous forest                                |     785 |        9.308 |
| ENVO:00000078 | farm                                                    |     696 |        8.252 |
| ENVO:01000888 | area of gramanoid or herbaceous vegetation              |     589 |        6.984 |
| ENVO:01000869 | area of scrub                                           |     510 |        6.047 |
| ENVO:00000209 | marine photic zone                                      |     289 |        3.427 |
| ENVO:00000022 | river                                                   |     281 |        3.332 |
| ENVO:00005801 | rhizosphere                                             |     229 |        2.715 |
| ENVO:00000148 | riffle                                                  |     198 |        2.348 |
| ENVO:00000011 | garden                                                  |     177 |        2.099 |
| ENVO:01000893 | area of woody wetland                                   |     165 |        1.956 |
| ENVO:00000469 | research facility                                       |     153 |        1.814 |
| ENVO:03600095 | stream run                                              |     152 |        1.802 |
| ENVO:01000855 | area of mixed forest                                    |     136 |        1.613 |
| ENVO:01000892 | area of cropland                                        |     107 |        1.269 |
| ENVO:01001057 | environment associated with a plant part or small plant |     100 |        1.186 |
| ENVO:03605007 | freshwater stream                                       |      92 |        1.091 |
| ENVO:00002131 | epilimnion                                              |      92 |        1.091 |
| ENVO:01000891 | area of pastureland or hayfields                        |      91 |        1.079 |



#### env_medium

- Total records: 8,434

- Unique terms: 32

- Terms after filter: 27

- **Imbalance Severity:** 🟡 HIGH

- Terms for 50% coverage: 2 (6.2%)

- Terms for 80% coverage: 4 (12.5%)

- Terms for 95% coverage: 13 (40.6%)


**Top 20 Most Frequent Terms:**


| term          | label                        |   count |   percentage |
|:--------------|:-----------------------------|--------:|-------------:|
| ENVO:00001998 | soil                         |    3812 |       45.198 |
| ENVO:00005749 | farm soil                    |    1812 |       21.484 |
| ENVO:00005802 | bulk soil                    |     858 |       10.173 |
| ENVO:03605001 | epilithon                    |     324 |        3.842 |
| ENVO:01000349 | root matter                  |     205 |        2.431 |
| ENVO:00002200 | sea ice                      |     159 |        1.885 |
| ENVO:04000007 | lake water                   |     153 |        1.814 |
| ENVO:00005791 | sterile water                |     152 |        1.802 |
| ENVO:00002042 | surface water                |     151 |        1.79  |
| ENVO:00002149 | sea water                    |     132 |        1.565 |
| ENVO:00002007 | sediment                     |     130 |        1.541 |
| ENVO:03605006 | stream water                 |      92 |        1.091 |
| ENVO:03605004 | epipsammon                   |      77 |        0.913 |
| ENVO:00005801 | rhizosphere                  |      64 |        0.759 |
| ENVO:01001001 | plant-associated environment |      56 |        0.664 |
| ENVO:00002261 | forest soil                  |      43 |        0.51  |
| ENVO:00002275 | technosol                    |      36 |        0.427 |
| ENVO:00005755 | field soil                   |      30 |        0.356 |
| ENVO:03605005 | epixylon                     |      24 |        0.285 |
| ENVO:00005781 | heat stressed soil           |      21 |        0.249 |



## Parent Class Analysis


### GOLD Parent Classes


#### env_broad_scale

| parent_term   | label                                          |   count |
|:--------------|:-----------------------------------------------|--------:|
| ENVO:01000254 | environmental system                           |   11083 |
| RO:0002577    | system                                         |   11083 |
| ENVO:01001110 | ecosystem                                      |   10908 |
| ENVO:00000428 | biome                                          |   10906 |
| ENVO:01000813 | astronomical body part                         |   10906 |
| ENVO:00000446 | terrestrial biome                              |    5634 |
| ENVO:01001790 | terrestrial ecosystem                          |    5634 |
| ENVO:01000997 | environmental system determined by a quality   |    5634 |
| ENVO:00002030 | aquatic biome                                  |    4721 |
| ENVO:01001787 | aquatic ecosystem                              |    4721 |
| ENVO:00000873 | freshwater biome                               |    2421 |
| ENVO:01001789 | freshwater ecosystem                           |    2421 |
| ENVO:01000320 | marine environment                             |    1652 |
| ENVO:01001788 | marine ecosystem                               |    1652 |
| ENVO:00000447 | marine biome                                   |    1652 |
| ENVO:01000252 | freshwater lake biome                          |    1071 |
| ENVO:01000174 | forest biome                                   |     977 |
| ENVO:01001243 | forest ecosystem                               |     977 |
| ENVO:01000253 | freshwater river biome                         |     883 |
| ENVO:01001828 | anthropised ecosystem                          |     359 |
| ENVO:01000313 | anthropogenic environment                      |     359 |
| ENVO:01000177 | grassland biome                                |     346 |
| ENVO:01001206 | grassland ecosystem                            |     346 |
| ENVO:01001000 | environmental system determined by an organism |     315 |
| ENVO:01000023 | marine pelagic biome                           |     291 |
| ENVO:01000024 | marine benthic biome                           |     282 |
| ENVO:01000219 | anthropogenic terrestrial biome                |     219 |
| ENVO:01000339 | polar biome                                    |     175 |
| ENVO:01001001 | plant-associated environment                   |     175 |
| ENVO:01001703 | polar environment                              |     175 |


✓ Expected parent `ENVO:00000428` found at rank 7 with count 10,906


#### env_local_scale

| parent_term   | label                           |   count |
|:--------------|:--------------------------------|--------:|
| ENVO:01000813 | astronomical body part          |    8826 |
| ENVO:01001784 | compound astronomical body part |    4953 |
| ENVO:01001479 | fluid astronomical body part    |    3186 |
| ENVO:01001477 | liquid astronomical body part   |    3186 |
| ENVO:01001476 | body of liquid                  |    2684 |
| ENVO:01000685 | water mass                      |    2684 |
| ENVO:00000063 | water body                      |    2683 |
| ENVO:01000408 | environmental zone              |    1786 |
| ENVO:00000191 | solid astronomical body part    |    1764 |
| ENVO:01001199 | terrestrial environmental zone  |    1755 |
| RO:0002577    | system                          |    1556 |
| ENVO:01001305 | vegetated area                  |    1439 |
| ENVO:01000617 | lentic water body               |    1416 |
| ENVO:01000254 | environmental system            |    1394 |
| ENVO:01001110 | ecosystem                       |    1382 |
| ENVO:01000281 | layer                           |    1246 |
| ENVO:00000029 | watercourse                     |    1210 |
| ENVO:01000618 | lotic water body                |    1210 |
| ENVO:00000023 | stream                          |    1184 |
| ENVO:01001886 | landform                        |    1161 |
| ENVO:00000020 | lake                            |     913 |
| ENVO:01001884 | surface landform                |     869 |
| ENVO:00000022 | river                           |     857 |
| ENVO:01000352 | field                           |     806 |
| ENVO:01001209 | wetland ecosystem               |     734 |
| ENVO:01001320 | fresh water body                |     731 |
| ENVO:00000021 | freshwater lake                 |     731 |
| ENVO:00000114 | agricultural field              |     721 |
| ENVO:00000070 | human construction              |     640 |
| ENVO:01001813 | construction                    |     640 |



#### env_medium

| parent_term   | label                              |   count |
|:--------------|:-----------------------------------|--------:|
| ENVO:00010483 | environmental material             |   10802 |
| ENVO:01000813 | astronomical body part             |    6684 |
| ENVO:00001998 | soil                               |    5086 |
| ENVO:02000140 | fluid environmental material       |    3063 |
| ENVO:01000815 | liquid environmental material      |    3053 |
| ENVO:00002006 | liquid water                       |    3025 |
| ENVO:01000060 | particulate environmental material |    1740 |
| ENVO:00002007 | sediment                           |    1624 |
| ENVO:00002010 | saline water                       |    1051 |
| ENVO:00002149 | sea water                          |     987 |
| ENVO:04000007 | lake water                         |     816 |
| ENVO:00002261 | forest soil                        |     695 |
| ENVO:00002243 | histosol                           |     584 |
| ENVO:00005774 | peat soil                          |     582 |
| ENVO:01000814 | solid environmental material       |     492 |
| ENVO:01000155 | organic material                   |     472 |
| ENVO:03605006 | stream water                       |     460 |
| ENVO:01000599 | river water                        |     460 |
| ENVO:00002011 | fresh water                        |     408 |
| ENVO:00002259 | agricultural soil                  |     375 |
| ENVO:00001995 | rock                               |     369 |
| ENVO:00002016 | sedimentary rock                   |     359 |
| ENVO:02000091 | coal                               |     359 |
| ENVO:00005802 | bulk soil                          |     317 |
| ENVO:03000033 | marine sediment                    |     289 |
| ENVO:00000546 | lake sediment                      |     236 |
| ENVO:00005781 | heat stressed soil                 |     226 |
| RO:0002577    | system                             |     190 |
| ENVO:01001110 | ecosystem                          |     190 |
| ENVO:01000254 | environmental system               |     190 |


✓ Expected parent `ENVO:00010483` found at rank 1 with count 10,802


### NCBI Parent Classes


#### env_broad_scale

| parent_term   | label                                        |   count |
|:--------------|:---------------------------------------------|--------:|
| ENVO:01000813 | astronomical body part                       |  355002 |
| RO:0002577    | system                                       |  265584 |
| ENVO:01000254 | environmental system                         |  254317 |
| ENVO:01001110 | ecosystem                                    |  234341 |
| ENVO:00000428 | biome                                        |  192890 |
| ENVO:01001787 | aquatic ecosystem                            |   77398 |
| ENVO:00002030 | aquatic biome                                |   76789 |
| ENVO:01000997 | environmental system determined by a quality |   68709 |
| ENVO:01001790 | terrestrial ecosystem                        |   67262 |
| ENVO:00010483 | environmental material                       |   63777 |
| ENVO:01001784 | compound astronomical body part              |   61703 |
| ENVO:00000446 | terrestrial biome                            |   55074 |
| ENVO:01000320 | marine environment                           |   41951 |
| ENVO:01001477 | liquid astronomical body part                |   40942 |
| ENVO:01001479 | fluid astronomical body part                 |   40942 |
| ENVO:00000063 | water body                                   |   40784 |
| ENVO:01001476 | body of liquid                               |   40784 |
| ENVO:01000685 | water mass                                   |   40784 |
| ENVO:01001788 | marine ecosystem                             |   40594 |
| ENVO:00000447 | marine biome                                 |   40215 |
| ENVO:01000313 | anthropogenic environment                    |   35904 |
| ENVO:01001828 | anthropised ecosystem                        |   27768 |
| ENVO:00001998 | soil                                         |   25095 |
| ENVO:01000617 | lentic water body                            |   24182 |
| ENVO:01001243 | forest ecosystem                             |   23824 |
| ENVO:01001789 | freshwater ecosystem                         |   23664 |
| ENVO:00000873 | freshwater biome                             |   23639 |
| ENVO:00000191 | solid astronomical body part                 |   20021 |
| ENVO:01000408 | environmental zone                           |   18177 |
| ENVO:01001199 | terrestrial environmental zone               |   18162 |


✓ Expected parent `ENVO:00000428` found at rank 6 with count 192,890


#### env_local_scale

| parent_term   | label                                        |   count |
|:--------------|:---------------------------------------------|--------:|
| ENVO:01000813 | astronomical body part                       |  295543 |
| RO:0002577    | system                                       |  120049 |
| ENVO:01001784 | compound astronomical body part              |  110049 |
| ENVO:01000254 | environmental system                         |  107860 |
| ENVO:00010483 | environmental material                       |   99959 |
| ENVO:01001110 | ecosystem                                    |   72488 |
| ENVO:01001477 | liquid astronomical body part                |   65102 |
| ENVO:01001479 | fluid astronomical body part                 |   65102 |
| ENVO:01000685 | water mass                                   |   57615 |
| ENVO:01001476 | body of liquid                               |   57615 |
| ENVO:00000063 | water body                                   |   57488 |
| ENVO:00000191 | solid astronomical body part                 |   42725 |
| ENVO:00001998 | soil                                         |   39295 |
| ENVO:02000140 | fluid environmental material                 |   33354 |
| ENVO:01001886 | landform                                     |   32801 |
| ENVO:01000313 | anthropogenic environment                    |   31976 |
| ENVO:01000815 | liquid environmental material                |   31952 |
| ENVO:00002006 | liquid water                                 |   31897 |
| ENVO:01000618 | lotic water body                             |   28610 |
| ENVO:00000029 | watercourse                                  |   28606 |
| ENVO:00000428 | biome                                        |   26043 |
| ENVO:01001884 | surface landform                             |   24585 |
| ENVO:01001828 | anthropised ecosystem                        |   24475 |
| ENVO:01000617 | lentic water body                            |   23661 |
| ENVO:01000408 | environmental zone                           |   23429 |
| ENVO:01001199 | terrestrial environmental zone               |   23404 |
| ENVO:01000997 | environmental system determined by a quality |   22229 |
| ENVO:00000077 | agricultural ecosystem                       |   21315 |
| ENVO:01001305 | vegetated area                               |   20963 |
| ENVO:01000352 | field                                        |   20684 |



#### env_medium

| parent_term   | label                              |   count |
|:--------------|:-----------------------------------|--------:|
| ENVO:00010483 | environmental material             |  328891 |
| ENVO:01000813 | astronomical body part             |  223439 |
| ENVO:00001998 | soil                               |  121039 |
| ENVO:02000140 | fluid environmental material       |  117405 |
| ENVO:01000815 | liquid environmental material      |  110502 |
| ENVO:00002006 | liquid water                       |  110195 |
| RO:0002577    | system                             |   48338 |
| ENVO:01000254 | environmental system               |   46509 |
| ENVO:00002010 | saline water                       |   41292 |
| ENVO:01001784 | compound astronomical body part    |   36286 |
| ENVO:01000155 | organic material                   |   34111 |
| ENVO:01001110 | ecosystem                          |   34068 |
| ENVO:01000060 | particulate environmental material |   31614 |
| ENVO:00002149 | sea water                          |   29773 |
| ENVO:00002007 | sediment                           |   28036 |
| ENVO:00002264 | waste material                     |   27587 |
| ENVO:01001479 | fluid astronomical body part       |   24128 |
| ENVO:01001477 | liquid astronomical body part      |   24088 |
| ENVO:01000685 | water mass                         |   22570 |
| ENVO:01001476 | body of liquid                     |   22570 |
| ENVO:00000063 | water body                         |   22567 |
| ENVO:02000019 | bodily fluid material              |   17275 |
| ENVO:02000022 | excreta material                   |   16144 |
| ENVO:00002003 | fecal material                     |   16088 |
| ENVO:01000618 | lotic water body                   |   15144 |
| ENVO:00000029 | watercourse                        |   15136 |
| ENVO:00002259 | agricultural soil                  |   12742 |
| ENVO:00000191 | solid astronomical body part       |   12195 |
| ENVO:01000408 | environmental zone                 |   11599 |
| ENVO:01001199 | terrestrial environmental zone     |   11599 |


✓ Expected parent `ENVO:00010483` found at rank 11 with count 328,891


### NMDC Parent Classes


#### env_broad_scale

| parent_term   | label                                          |   count |
|:--------------|:-----------------------------------------------|--------:|
| ENVO:01000254 | environmental system                           |    8378 |
| RO:0002577    | system                                         |    8378 |
| ENVO:00000428 | biome                                          |    8238 |
| ENVO:01001110 | ecosystem                                      |    8238 |
| ENVO:01000813 | astronomical body part                         |    8238 |
| ENVO:01000997 | environmental system determined by a quality   |    6900 |
| ENVO:00000446 | terrestrial biome                              |    6808 |
| ENVO:01001790 | terrestrial ecosystem                          |    6808 |
| ENVO:01001705 | temperate environment                          |    1830 |
| ENVO:01001831 | temperate biome                                |    1830 |
| ENVO:01000175 | woodland biome                                 |    1824 |
| ENVO:01000221 | temperate woodland biome                       |    1820 |
| ENVO:01001787 | aquatic ecosystem                              |    1287 |
| ENVO:00002030 | aquatic biome                                  |    1287 |
| ENVO:00000873 | freshwater biome                               |     995 |
| ENVO:01001789 | freshwater ecosystem                           |     995 |
| ENVO:01001828 | anthropised ecosystem                          |     887 |
| ENVO:01000313 | anthropogenic environment                      |     887 |
| ENVO:01000219 | anthropogenic terrestrial biome                |     762 |
| ENVO:01000253 | freshwater river biome                         |     750 |
| ENVO:01001244 | cropland ecosystem                             |     712 |
| ENVO:01000245 | cropland biome                                 |     712 |
| ENVO:01000320 | marine environment                             |     292 |
| ENVO:00000447 | marine biome                                   |     292 |
| ENVO:01001788 | marine ecosystem                               |     292 |
| ENVO:01000033 | oceanic pelagic zone biome                     |     291 |
| ENVO:01000023 | marine pelagic biome                           |     291 |
| ENVO:01000036 | oceanic mesopelagic zone biome                 |     290 |
| ENVO:01001000 | environmental system determined by an organism |     265 |
| ENVO:01000252 | freshwater lake biome                          |     153 |


✓ Expected parent `ENVO:00000428` found at rank 8 with count 8,238


#### env_local_scale

| parent_term   | label                                      |   count |
|:--------------|:-------------------------------------------|--------:|
| ENVO:01000813 | astronomical body part                     |    7472 |
| ENVO:01001199 | terrestrial environmental zone             |    3621 |
| ENVO:01000408 | environmental zone                         |    3621 |
| ENVO:01001305 | vegetated area                             |    2836 |
| ENVO:01000352 | field                                      |    1828 |
| ENVO:00000114 | agricultural field                         |    1824 |
| ENVO:01001784 | compound astronomical body part            |    1364 |
| RO:0002577    | system                                     |    1132 |
| ENVO:01000254 | environmental system                       |    1132 |
| ENVO:01001110 | ecosystem                                  |    1113 |
| ENVO:01001477 | liquid astronomical body part              |    1043 |
| ENVO:01001479 | fluid astronomical body part               |    1043 |
| ENVO:01000843 | area of evergreen forest                   |    1041 |
| ENVO:01000816 | area of deciduous forest                   |     785 |
| ENVO:01001828 | anthropised ecosystem                      |     738 |
| ENVO:01000313 | anthropogenic environment                  |     738 |
| ENVO:00000077 | agricultural ecosystem                     |     696 |
| ENVO:00000078 | farm                                       |     696 |
| ENVO:01000888 | area of gramanoid or herbaceous vegetation |     680 |
| ENVO:00000106 | grassland area                             |     680 |
| ENVO:01001308 | hydroform                                  |     645 |
| ENVO:01001293 | bush area                                  |     593 |
| ENVO:01000869 | area of scrub                              |     510 |
| ENVO:00000063 | water body                                 |     504 |
| ENVO:01001476 | body of liquid                             |     504 |
| ENVO:01000685 | water mass                                 |     504 |
| ENVO:01000281 | layer                                      |     456 |
| ENVO:01001678 | fluid layer                                |     413 |
| ENVO:01000325 | aquatic layer                              |     401 |
| ENVO:01001273 | liquid layer                               |     401 |



#### env_medium

| parent_term   | label                              |   count |
|:--------------|:-----------------------------------|--------:|
| ENVO:00010483 | environmental material             |    8314 |
| ENVO:01000813 | astronomical body part             |    6992 |
| ENVO:00001998 | soil                               |    6670 |
| ENVO:00005749 | farm soil                          |    1812 |
| ENVO:00005802 | bulk soil                          |     858 |
| ENVO:01000815 | liquid environmental material      |     693 |
| ENVO:02000140 | fluid environmental material       |     693 |
| ENVO:00002006 | liquid water                       |     693 |
| ENVO:01000155 | organic material                   |     662 |
| ENVO:01000156 | biofilm material                   |     456 |
| ENVO:03605000 | periphytic biofilm                 |     456 |
| ENVO:03605001 | epilithon                          |     324 |
| ENVO:01000349 | root matter                        |     205 |
| ENVO:01000277 | water ice                          |     159 |
| ENVO:01001557 | water-body-derived ice             |     159 |
| ENVO:00002200 | sea ice                            |     159 |
| ENVO:01000814 | solid environmental material       |     159 |
| ENVO:01001125 | ice                                |     159 |
| ENVO:04000007 | lake water                         |     153 |
| ENVO:00005791 | sterile water                      |     152 |
| ENVO:00002042 | surface water                      |     151 |
| ENVO:00002149 | sea water                          |     132 |
| ENVO:00002010 | saline water                       |     132 |
| ENVO:00002007 | sediment                           |     130 |
| ENVO:01000060 | particulate environmental material |     130 |
| RO:0002577    | system                             |     120 |
| ENVO:01000254 | environmental system               |     120 |
| ENVO:03605006 | stream water                       |     105 |
| ENVO:03605004 | epipsammon                         |      77 |
| ENVO:01001110 | ecosystem                          |      64 |


✓ Expected parent `ENVO:00010483` found at rank 1 with count 8,314


## ML Strategy Recommendations


1. **Start with NMDC** if available (typically cleanest metadata)

2. **Use parent class grouping** for terms below your threshold

3. **Consider separate models** per dataset if vocabularies differ significantly

4. **For combined training**, normalize term usage across datasets first

5. **Monitor class imbalance** - severe imbalance may require min_pct ≥ 1.0%
