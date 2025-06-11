# STARsim

STARsim is a simulation model for air traffic in the vicinity of the airport, focusing on the arrival traffic.

## Description

The model was created to evaluate and test concepts in arrival traffic management, in particular control strategies 
and STAR procedures topologies. The simulations happen in the environment automatically created from the data on 
airport and STAR procedures provided by the user. Outcome of simulation comes in the form of statistics regarding 
network performance, aircraft positional data or as a visualisation of a run. The implementation is mainly written in 
Python with small bits of JavaScript for visualisations.

## Getting Started

### Dependencies

* Python v3.9.12 or newer

### Setting up the environment
1. Clone the repository into a working directory on your machine
2. Create a virtual environment and install required packages:
    * If using conda, use file environment.yaml  with:
      ```
      conda env create -f environment.yml
      ```
    * If using pip, use requirements.txt:
       ```
      cd YourWorkingDirectory/STARsim
      python -m venv STARsimEnv 
      source STARsimEnv/bin/activate
      pip install -r requirements.txt --user
      ```

      

### Executing program
  1. Change directory to the project root
  2. Run STARsim_starter.py, passing the appropriate arguments as flags
  ```
  python STARsim/STARsim_starter.py -<flag><arg>
  ```
###Examples:
* To run five simulations in one of the setups used in my Master's Thesis, run (from project root directory):
    ```
    python STARsim/STARsim_starter.py -T 01:30:00 -D data_c -N 5 -v 2,4 -s stats -i initial.csv
  ```
 * To  run five simulations with random initial conditions
    ```
    python STARsim/STARsim_starter.py -T 01:30:00 -D data_c -N 5 -v 2 -n 5 -s stats
     ```
These statistics from the runs will be saved to a file stats.pickle in the wd.
THE VISUALISING PART TAKES A WHILE, SO WHEN TERMINAL STOPS PRINTING, GIVE IT A MINUTE! 
## Help

For info on arguments, run:
```
  python STARsim/STARsim_starter.py -h
```



## Authors


[Kacper Kartus](manofrushmore@proton.me)

## Version History

* 0.1
    * Initial Release
