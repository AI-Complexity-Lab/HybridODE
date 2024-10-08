# Simulate Data

This script `simulate_data.py` contains functions for generating synthetic epidemiological data based on the SEIR-HD model, which is an expert ODE model used for simulating the spread of infectious diseases. The model generates data for various disease compartments, including asymptomatic and pre-symptomatic stages, and factors in interventions like mobility changes and mask controls through beta decay.

## Functions

### 1. `generate_initial_conditions()`
This function generates initial conditions for the simulation. It creates 25 different sets of initial conditions representing various population compartments at the start of the simulation. These compartments include:
- **Population (Total Population)**: The total number of individuals in the population.
- **Susceptible**: The number of individuals who are susceptible to the disease.
- **Exposed**: The number of individuals who have been exposed to the virus but are not yet infectious.
- **Infectious Asymptomatic**: The number of individuals who are infected but not showing symptoms and can transmit the disease.
- **Infectious Pre-symptomatic**: The number of individuals who are infected and in the pre-symptomatic stage before showing any symptoms but are capable of spreading the virus.
- **Infectious Mild**: The number of individuals with mild symptoms.
- **Infectious Severe**: The number of individuals with severe symptoms.
- **Hospitalized Recovered**: The number of individuals who were hospitalized but recovered.
- **Recovered**: The number of individuals who have recovered from the infection.
- **Deceased**: The number of individuals who have died from the disease.

### 2. `generate_beta_schedule()`
This function generates a schedule for the transmission rate (`beta`). The transmission rate (`beta`) represents the likelihood of disease transmission between susceptible and infectious individuals. In this model, `beta` decay is used to represent the impact of interventions like mobility restrictions or mask mandates. Over time, `beta` decays to simulate the effects of treatments and policy changes in reducing the disease's transmission rate.

### 3. `generate_inference_csv()`
This function simulates epidemiological data using the SEIR-HD model (a system of expert ODEs). The function generates 100 different combinations of:
- **Initial conditions**: Generated using `generate_initial_conditions()`.
- **Incubation rate (`Ca`)**: The rate at which exposed individuals transition to the infectious stage. Higher values indicate a shorter incubation period.
- **Recovery rate (`Delta`)**: The rate at which infected individuals recover and move into the recovered compartment.

For each combination of initial conditions, `Ca`, and `Delta`, the simulation runs for 500 days. The generated data is saved in CSV format and can be used for further analysis, inference, or model fitting.

## Usage
To use this script, simply call the respective functions to generate the required datasets:
```bash
python simulate_data.py
```
This will create a dataset that includes epidemiological data with different combinations of initial conditions, incubation rates, and recovery rates over a period of 500 days.

## Output
The output of this script includes:
- 25 different sets of initial conditions for the simulation.
- 100 different simulations using varying combinations of `Ca` and `Delta` parameters.
- CSV files containing the simulation data for further analysis.

