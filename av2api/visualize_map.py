from av2.datasets.motion_forecasting.viz import scenario_visualization
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario
from av2.map.map_api import ArgoverseStaticMap

from argparse import Namespace
from pathlib import Path

import pandas as pd
import os

import visualization
import submission

from av2.datasets.motion_forecasting import scenario_serialization

# static_map_path: Path to the JSON file containing map data. The file name must match
# the following pattern: "log_map_archive_{log_id}.json".
# path to where the logs live
dataroot = "/home/vgrwbx/workspace/OL_trajectoryprediction/data/val/raw"
# dataroot = '/home/vgrwbx/data/datasets/test'

# unique log identifier
# log_id = '0a0af725-fbc3-41de-b969-3be718f694e2'
log_id='705bf89c-7464-4f36-bec3-ee61d2b2f174'
# log_id = "adc7a713-efdf-40db-a952-e55f1fcda675"

ol_path = '/home/vgrwbx//workspace/OL_trajectoryprediction/submission_05.parquet'

args = Namespace(**{"dataroot": Path(dataroot), "log_id": Path(log_id)})

log_map_dirpath = Path(args.dataroot) / args.log_id 

print(log_map_dirpath)

scenario_static_map = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

scenario = pd.read_parquet(os.path.join(log_map_dirpath, f'scenario_{log_id}.parquet'))

test_scenario = scenario_serialization.load_argoverse_scenario_parquet(
        os.path.join(log_map_dirpath, f'scenario_{log_id}.parquet')
    )

ol_submission = submission.ChallengeSubmission.from_parquet(Path(ol_path))

visualization.plot_map(scenario_static_map)

visualization.plot_map_history(scenario_static_map,test_scenario)

visualization.plot_target_history(test_scenario)

visualization.plot_history_predictions(scenario_static_map, test_scenario, ol_submission)