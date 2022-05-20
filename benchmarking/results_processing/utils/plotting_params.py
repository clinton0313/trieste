
import os
from trieste.objectives.single_objectives import *

RESULTS_DIR = os.path.realpath(os.path.join("..", "..", "results"))

#Dict of keys: directory name; name: model prefix used in results; label: label for plotting.

OBJECTIVE_DICT = {
    "michal2": "Michalwicz-2",
    "scaled_branin": "Scaled Branin-2",
    "hartmann6": "Hartmann-6",
    "hartmann3": "Hartmann-3",
    "goldstein2": "Log Goldstein-Price-2",
    "shekel4": "Shekel-4",
    "rosenbrock4": "Rosenbrock-4",
    "ackley5": "Ackley-5",
    "michal5": "Michalwicz-5",
    "michal10": "Michalwicz-10",
    "dropw2": "Dropwave-2",
    "eggho2": "Eggholder-2"
}

OBJ_MIN_DICT = {
    "michal2": MICHALEWICZ_2_MINIMUM,
    "scaled_branin": BRANIN_MINIMUM, 
    "hartmann6": HARTMANN_6_MINIMUM, 
    "goldstein2": LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM, 
    "hartmann3": HARTMANN_3_MINIMUM, 
    "shekel4": SHEKEL_4_MINIMUM, 
    "rosenbrock4": ROSENBROCK_4_MINIMUM, 
    "ackley5": ACKLEY_5_MINIMUM, 
    "dropw2": DROPWAVE_MINIMUM, 
    "eggho2": EGGHOLDER_MINIMUM,
    "michal5": MICHALEWICZ_5_MINIMUM,
    "michal10": MICHALEWICZ_10_MINIMUM
}

OUTPUT_RANGE_DICT = {
    "michal2": 1.801303345635689, 
    "scaled_branin": 5.918577123175378, 
    "hartmann6": 3.303679123380341, 
    "goldstein2": 5.246159889348281, 
    "hartmann3": 3.862716226272016, 
    "shekel4": 9.727798821861356, 
    "rosenbrock4": 0.17934849140777742, 
    "ackley5": 19.161816848081262, 
    "dropw2": 0.9999982299815201, 
    "eggho2": 2008.6400742211358,
    "michal5": 1.8, #PLACEHOLDERS HERE
    "michal10": 1.8, #PLACEHOLDERS HERE
}

ACQUISITION_DICT = {
    "ei": "Expected Improvement",
    "ts": "Thompson Sampling"
}

MODELS_DICT = {
    "der": {
        "name": "new_der_log",
        "label": "Deep Evidential",
    },
    "deup": {
        "name": "deup",
        "label": "Direct Epistemic",
    },
    "de": {
        "name": "de",
        "label": "Deep Ensembles",
    },
    "gpr": {
        "name": "gpr",
        "label": "GPR",
    },
    "mc": {
        "name": "mc",
        "label": "MC Dropout",
    },
    "random": {
        "name": "random",
        "label": "Random Sampling",
    },
}

COLOR_DICT = dict(zip(MODELS_DICT.keys(), matplotlib.colors.TABLEAU_COLORS.keys()))