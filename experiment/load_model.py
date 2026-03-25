import json
import veritas
import pickle

all_datasets = [
    "Electricity",
    "MiniBooNE",
    "Jannis",
    "Credit",
    "California",
    "CompasTwoYears",
    "Vehicle",
    "Spambase",
    "Phoneme",
    "Adult",
    "Ijcnn1",
    "Mnist[2v4]",
    "DryBean[6vRest]",
    "Volkert[2v7]"]

with open('experiment/pareto_fronts_LOP.pkl', 'rb') as f:
    pareto_fronts = pickle.load(f)

print(pareto_fronts)

# Load all         
lop_compressed_models_on_front = {}
with open('experiment/results/xgb_classification_saved.txt', 'r') as file: #Load in the XGB classification results
    for line in file:
        if not line.startswith('{'):
            continue
        line_dict = json.loads(line.strip())
        pareto_fronts_ds = pareto_fronts[line_dict['dname']]
        if pareto_fronts_ds[(pareto_fronts_ds['n_estimators'].astype(float) == float(line_dict['params']['n_estimators'])) &
                        (pareto_fronts_ds['max_depth'].astype(float) == float(line_dict['params']['max_depth'])) &
                        (pareto_fronts_ds['learning_rate'].astype(float) == float(line_dict['params']['learning_rate']))]['on_front'].values[0] == False:
             continue
        key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
        lop_compressed_models_on_front[key] = line_dict['refinements'][3]['model_json']


# Access all pareto front models for one dataset
dname = "Phoneme"
pareto_models_phoneme = {k: veritas.AddTree.from_json(v) for k, v in lop_compressed_models_on_front.items() if k.startswith(dname)}

print(f"Loaded {len(pareto_models_phoneme)} pareto front models for dataset {dname}.")

# Access a specific model
key = "Phoneme_10_4_0.1_0"  # Example key for Phoneme dataset, 10 estimators, max depth 4, learning rate 0.1, fold 0 before compression. Careful, not all combinations of hyperparameters are on the pareto front.
at = veritas.AddTree.from_json(lop_compressed_models_on_front[key])


