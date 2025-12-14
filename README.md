# SZ4D Open DAS Data for Subduction Zone Environments

## Overview
This half-day mini-workshop aims to introduce participants to publicly available DAS datasets and open-source tools, demonstrate analysis techniques relevant to subduction zone science, foster collaboration and data sharing to enhance reproducibility within the SZ4D community, and build connections among early-career and established researchers.

The event will take place at the Tulane River & Coastal Center (1370 Port of New Orleans Pl, New Orleans, LA 70130), located one block south of the Convention Center ([map](https://www.google.com/maps/d/u/1/edit?mid=1UC7Jxo5ibAlLGRE-w5grOP8jwSccRHc&usp=sharing))

## Agenda
|       Time       | Event |
|------------------|-------|
|  8:30 -  9:00 AM | Welcome coffee |
|  9:00 -  9:20 AM | Introduction from the Conveners |
|  9:20 -  9:35 AM | Getting set up with the computing environment - Yiyu Ni |
|||
| | **Hands on Session: DAS in a Subduction Zone Forearc Basin: Cook Inlet, AK** |
|  9:35 -  9:50 AM | Reading and plotting DAS data and earthquake signals - Zoe Krauss ([notebook](./notebooks/session1/eqplotting_cidas_AGU2025.ipynb))|
|  9:50 - 10:20 AM | Wavefield reconstruction with machine learning - Yiyu Ni ([notebook](./notebooks/session2/wavefield_cidas_AGU2025.ipynb)) |
| 10:20 - 10:35 AM | **Coffee Break** |
| | **Hands on Session: DAS at a Subduction Zone Volcano: Mount Rainier, WA** |
| 10:35 - 11:05 AM | Template matching and seismic detection - Verónica Gaete-Elgueta ([notebook](./notebooks/session3/template-matching/main_template_matching_moradas_AGU2025.ipynb)) |
| | **Hands on Session: DAS in a Subduction Zone Accretionary Prism: the Ocean Observatory Initiative Regional Cabled Array, OR** |
| 11:05 - 11:35 AM | Low-frequency oceanic waves, marine mammals, noise sources - Ethan Williams ([notebook](./notebooks/session4/ooi_ocean_waves.ipynb)) |
| 11:35 - 12:05 PM | Self-supervised machine learning and earthquake monitoring - Qibin Shi ([notebook](./notebooks/session5/denoise_pick_ooidas_AGU2025.ipynb)) |
| | **Hands on Session: DAS in Subduction Zone in Urban Areas, Seattle, WA** |
| 12:05 - 12:30 PM | Environmental noise correlation - Manuela Köpfli ([notebook](./notebooks/session6/noisepy_seadasn_AGU2025.ipynb)) |
|||
| 12:30 PM | **Workshop adjourned** |

## Organizers 
- Verónica Gaete-Elgueta (University of Washington) 
- Manuela Köpfli (University of Washington)
- Qibin Shi (Rice University)
- Zoe Krauss (University of Washington)
- Yiyu Ni (University of Washington) 
- Ethan Williams (UC Santa Cruz)
- Brad Lipovsky (University of Washington)

## Get Started
We will use the EarthScope GeoLab computing environment for this workshop. If you do not have access to GeoLab, please sign up at https://www.earthscope.org/data/geolab/.

When launching a GeoLab Jupyter hub, please select the "GeoLab" environment. Please use an allocation with more than 7 GB RAM. After launching the jupyter hub, copy and run the scripts below in a terminal. 

```bash
wget https://raw.githubusercontent.com/uwfiberlab/sz4d_das_workshop/refs/heads/main/configure.sh -O configure.sh
chmod +x configure.sh
./configure.sh
```

Wait until the configuration script finishes, then navigate into sz4d_das_workshop -> notebooks, and select the session notebook. When running the notebook, select the **workshop** kernel.
