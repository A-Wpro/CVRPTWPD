
<!-- ABOUT THE PROJECT -->
## About The Project

Context : 
optimisation helicpote routing tepdk
using HPC

Position des rigs/platform dans GPS.json




## Built With

* [![Python][python.py]][python-url]
* [![Or-Tools][ortools.py]][ortools-url]
 

## Prerequisites



* Or-tools is needed : 
  ```sh
  python -m pip install --upgrade --user ortools
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Clone this repo, modify config.json and run main.py or other relevant files.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Roadmap

- [X] Spliting code in files
- [X] Adding more CONST to config.json
- [X] Uploading tools
    - [X] Combi
- [X] Parqueting
    - [X] Output
    - [X] Combi
- [ ] Deep learning model for Cap/Sclice couple
    - [X] DL NN
    - [ ] RL
        - [X] Impact DL with solver
        - [ ] Cost Function
- [ ] Better meta heuristic
    - [X] Heuristic test
    - [ ] Implement
- [ ] Postprocessing 
- [ ] Deploy heurisitcs
- [ ] Creating folder for plot and adding tool for plot
- [ ] Statistical study
    - [X] Create enough solution
    - [ ] Graph and stats
- [ ] POC UX

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## List of hyper parameters
Hyper paramters are varaible that once define don't change during the whole process however those paramters can have a big impact on solution, here a list : 

* Every AddDimension paramters
* Number of vehicule
* 1st solution
* Heurtisct in solver
* Timer of wall (30 sec)
* Caping (Should be define by DL/RL)
* Sclicing (Should be define by DL/RL)
* Other heuristic paramters


## External docs
https://en.wikipedia.org/wiki/Guided_local_search

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
 
[ortools.py]: https://img.shields.io/badge/ortools-000000?style=for-the-badge&logo=google&logoColor=white
[ortools-url]: https://developers.google.com/optimization

[python.py]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
 