# from av2.datasets.motion_forecasting.viz import scenario_visualization
from av2.map.map_api import ArgoverseStaticMap

# from argparse import Namespace
from pathlib import Path

# import pandas as pd
import os
import numpy as np
from av2.utils.typing import NDArrayFloat

from argparse import ArgumentParser
import visualization #custom visualization api
from av2.datasets.motion_forecasting import scenario_serialization

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

from av2.datasets.motion_forecasting.data_schema import TrackCategory
from av2.datasets.motion_forecasting.eval import metrics


if __name__ == '__main__':

    parser = ArgumentParser()
    # static_map_path: Path to the JSON file containing map data. The file name must match
    # the following pattern: "log_map_archive_{log_id}.json".
    # path to where the logs live
    parser.add_argument('--dataroot', type=str, default="/home/vgrwbx/workspace/OL_trajectoryprediction/data/val/raw")
    parser.add_argument('--log_id', type=str, default="aa95458f-d095-4cee-b774-f075c3f1d2c2") # unique log identifier
    parser.add_argument('--submission_file_path', type=str, default="~/workspace/OL_trajectoryprediction/metrics_files/submission_val.parquet") # path of the submission
    parser.add_argument('--save_path', type=str, default="/home/vgrwbx/workspace/OL_trajectoryprediction/videos/test") # path where to save the visualization
    parser.add_argument('--timestep', type=int, default=30) # timestep
    args = parser.parse_args()

    log_map_dirpath = Path(args.dataroot) / args.log_id 
    
    # load map
    scenario_static_map = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

    # load scenario
    # scenario = pd.read_parquet(os.path.join(log_map_dirpath, f'scenario_{args.log_id}.parquet'))
    scenario = scenario_serialization.load_argoverse_scenario_parquet(
        os.path.join(log_map_dirpath, f'scenario_{args.log_id}.parquet')
    )

    # load challenge submission predictions, note that they might be on a different dataset!
    submission = ChallengeSubmission.from_parquet(Path(args.submission_file_path))


    # print(submission)

    #visualization.visualize_scenario(scenario, scenario_static_map, Path(args.save_path))


    visualization.visualize_predictions(scenario,submission, scenario_static_map, Path(args.save_path), timestep=109)



    #compute metrics
    # # for each element in the submission file get predicted trajectories and gt
    outer_key = scenario.scenario_id
    coordinates_array = [] 
    probabilities_array = [] 
    track_predictions = []  

    predictions = submission.predictions[scenario.scenario_id]  

    print('Scenario id: ', scenario.scenario_id)

    for track in scenario.tracks:
        if track.category == TrackCategory.FOCAL_TRACK:
            # Get actor trajectory and heading ground truth -> last 6 seconds
            gt_trajectory: NDArrayFloat = np.array(
                [list(object_state.position) for object_state in track.object_states if object_state.timestep >= 50]
            )
            actor_headings_gt: NDArrayFloat = np.array(
                [object_state.heading for object_state in track.object_states if object_state.timestep >= 50]
            )
            if len(predictions) == 0:
                print(f'{scenario.scenario_id} does not exist in the submission file')
            else:  
                print('Track id: ',track.track_id) 
                forecasted_trajectories = predictions[track.track_id][0] 

    # forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
    # gt_trajectory: (N, 2) ground truth trajectory.
    # print(forecasted_trajectories.shape)
    # print(gt_trajectory.shape)
    ade = metrics.compute_ade(forecasted_trajectories, gt_trajectory) 
    fde = metrics.compute_fde(forecasted_trajectories, gt_trajectory) 
    print('-------------------------METRICS (computed using av2-api)-----------------------')
    print('ADE', ade, ' - minADE:', min(ade))
    print('FDE', fde, ' - minFDE:', min(fde))

    # for inner_key, inner_value in outer_value.items():
    #     coordinates_array = inner_value[0]
    #     probabilities_array = inner_value[1]

        # print(f'scenario_id {outer_key}, track_id: {inner_key}')
        # print(f'Coordinates Array: {coordinates_array}')
        # print(f'Probabilities Array: {probabilities_array}')
