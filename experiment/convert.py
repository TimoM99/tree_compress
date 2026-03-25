import json
import pandas as pd
results_oc_calcs = {}
datasets = set()
with open('experiment/results/funky_norm.txt', 'r') as file:
    for line in file:
        if not line.startswith('{'):
            continue
        line_dict = json.loads(line.strip())
        key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
        results_oc_calcs[key] = line_dict
        datasets.add(line_dict['dname'])

        # Create a CSV with SAT results and time

        csv_data = []
        for key, result in results_oc_calcs.items():
            dname = result['dname']
            n_estimators = result['params']['n_estimators']
            max_depth = result['params']['max_depth']
            learning_rate = result['params']['learning_rate']
            fold = result['fold']
            
            # Extract SAT result and time
            try:
                sat_result = result['funky_norm']['verification_results']['sat']
                verification_time = result['funky_norm']['verification_results']['time']
            except KeyError:
                sat_result = 'N/A'
                verification_time = 'N/A'

            csv_data.append({
                'key': key,
                '#SAT': sat_result,
                'time': verification_time
            })

        # Convert to DataFrame and save to CSV
        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv('funky_norm_results.csv', index=False)
        print(f"CSV saved with {len(df_csv)} rows")
        print(df_csv.head())