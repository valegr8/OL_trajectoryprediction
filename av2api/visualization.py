# Visualization Script, modification of av2 api
# Valeria Grotto 20010126-6021 


"""Visualization utils for Argoverse MF scenarios."""

import io
import math
from pathlib import Path
from typing import Final, List, Optional, Sequence, Set, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image as img
from PIL.Image import Image

from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType, TrackCategory

from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt


import matplotlib.colors as mcolors

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_PREDICTION_COLOR: Final[str] = "#ffffff"
_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}


def visualize_scenario(scenario: ArgoverseScenario, scenario_static_map: ArgoverseStaticMap, save_path: Path) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """
    # Build each frame for the video
    frames: List[Image] = []
    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
        _, ax = plt.subplots()

        # Plot static map elements and actor tracks
        _plot_static_map_elements(scenario_static_map)
        cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
        if cur_plot_bounds:
            plot_bounds = cur_plot_bounds

        # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
        plt.xlim(plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M)
        plt.ylim(plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M)
        plt.gca().set_aspect("equal", adjustable="box")

        # Minimize plot margins and make axes invisible
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Save plotted frame to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

    # Write buffered frames to MP4V-encoded video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")

    print('Saved in: ', vid_path)
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

def visualize_predictions(scenario: ArgoverseScenario, submission: ChallengeSubmission,  scenario_static_map: ArgoverseStaticMap, save_path: Path, timestep: int = None, online_learning = False):
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        submission: Argoverse challenge submission format, it stores the predicted trajectories
        scenario_static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """
    # submission_dict: Dict[str, ScenarioPredictions] = defaultdict(lambda: defaultdict(tuple))
    # for prediction in submission_dict.items():
    #     if(prediction['scenario_id']==scenario['scenario_id']):
    #         print('Found scenario: ', scenario['scenario_id'])
    #         break


    #for outer_key, outer_value in submission.predictions.items():

    coordinates_array = [] 
    probabilities_array = [] 
    
    predictions_val = submission.predictions[scenario.scenario_id]    
    if len(predictions_val) == 0:
        print(f'{scenario.scenario_id} does not exist in the submission file')

    fig, ax = plt.subplots()
    if timestep == None: 
        # video
        # Build each frame for the video
        frames: List[Image] = []
        plot_bounds: _PlotBounds = (0, 0, 0, 0)

        for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
            _, ax = plt.subplots()

            # Plot static map elements and actor tracks
            _plot_static_map_elements(scenario_static_map)
            cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
            if cur_plot_bounds:
                plot_bounds = cur_plot_bounds

            i = 0
            #plot prediction
            if online_learning:
                # Get an iterator for the dictionary items
                iterator = iter(predictions_val.items())
                try:
                    # Get the next prediction item
                    next_key, next_val = next(iterator)
                except StopIteration:
                    # Handle the case when there are no more items
                    next_key = None
                exit_check = False
                for timestep_key, timestep_val in predictions_val.items():
                    # Access the current prediction item
                    try:
                        # Get the next prediction item
                        next_key, next_val = next(iterator)
                    except StopIteration:
                        # Handle the case when there are no more items
                        next_key = None
                    if next_key==None or (timestep_key >= timestep and timestep<next_key):
                        for track_id, track_val in timestep_val.items():
                            i +=1
                            coordinates_array = track_val[0]
                            probabilities_array = track_val[1]

                            #plot prediction
                            _plot_polylines(coordinates_array, style='--',line_width=1, probabilities=probabilities_array)
                            exit_check = True
                        
                        if exit_check:
                            break
            else:
                for inner_key, inner_value in predictions_val.items():
                    i +=1
                    coordinates_array = inner_value[0]
                    probabilities_array = inner_value[1]

                    #plot prediction
                    _plot_polylines(coordinates_array, style='--',line_width=1, probabilities=probabilities_array)

            # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
            plt.xlim(plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M)
            plt.ylim(plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M)
            plt.gca().set_aspect("equal", adjustable="box")

            # Minimize plot margins and make axes invisible
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            # Save plotted frame to in-memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")

            plt.close()
            buf.seek(0)
            frame = img.open(buf)
            frames.append(frame)
            
        # Write buffered frames to MP4V-encoded video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")

        print('Saved in: ', vid_path)
        video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
        for i in range(len(frames)):
            frame_temp = frames[i].copy()
            video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
        video.release()
        
    else:
        # Plot static map elements and actor tracks
        _plot_static_map_elements(scenario_static_map)
        _plot_actor_tracks(ax, scenario, timestep)

        i = 0
        #plot prediction
        if online_learning:
            # Get an iterator for the dictionary items
            iterator = iter(predictions_val.items())
            try:
                # Get the next prediction item
                next_key, next_val = next(iterator)
            except StopIteration:
                # Handle the case when there are no more items
                next_key = -1
            exit_check = False
            for timestep_key, timestep_val in predictions_val.items():
                # Access the current prediction item
                try:
                    # Get the next prediction item
                    next_key, next_val = next(iterator)
                except StopIteration:
                    # Handle the case when there are no more items
                    next_key = -1

                if next_key==-1 or (timestep_key >= timestep and timestep<next_key):
                    for track_id, track_val in timestep_val.items():
                        i +=1
                        coordinates_array = track_val[0]
                        probabilities_array = track_val[1]

                        #plot prediction
                        _plot_polylines(coordinates_array, style='--',line_width=1, probabilities=probabilities_array)
                        exit_check = True
                    
                    if exit_check:
                        break
        else:
            for inner_key, inner_value in predictions_val.items():
                i +=1
                coordinates_array = inner_value[0]
                probabilities_array = inner_value[1]

                #plot prediction
                _plot_polylines(coordinates_array, style='--',line_width=1, probabilities=probabilities_array)

        # plt.show()
        
        # Save the figure
        fig_path = str(save_path.parents[0] / f"{save_path.stem}.png")
        fig.savefig(fig_path, dpi=300)
        plt.close()  # Close the plot to free memory
        print(fig_path)

        return fig_path

def __probability_to_color(probability):
    # Create a colormap from yellow to red
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["yellow", "red"])
    # Normalize the probability value between 0 and 1
    normalized_probability = mcolors.Normalize(vmin=0, vmax=1)(probability)
    # Get the color corresponding to the normalized probability
    color = cmap(normalized_probability)
    # Convert the RGBA color to a hexadecimal string
    hex_color = mcolors.rgb2hex(color)
    return hex_color



def _plot_static_map_elements(static_map: ArgoverseStaticMap, show_ped_xings: bool = False) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines([ped_xing.edge1.xyz, ped_xing.edge2.xyz], alpha=1.0, color=_LANE_SEGMENT_COLOR)

def plot_map(static_map: ArgoverseStaticMap, show_ped_xings: bool = False) -> None:
    _plot_static_map_elements(static_map, show_ped_xings)
    plt.show()

def plot_map_history(static_map: ArgoverseStaticMap, scenario: ArgoverseScenario, show_ped_xings: bool = False) -> None:
    _, ax = plt.subplots()
    _plot_static_map_elements(static_map, show_ped_xings)
    _plot_actor_tracks(ax, scenario, 50)
    plt.show()

def plot_history_predictions(static_map: ArgoverseStaticMap, scenario: ArgoverseScenario, submission: ChallengeSubmission, online_learning:bool = True, timestep: int = 50, show_ped_xings: bool = False) -> None:
    _, ax = plt.subplots()
    history = 50
    coordinates_array = [] 
    probabilities_array = [] 
    
    predictions_val = submission.predictions[scenario.scenario_id]    
    if len(predictions_val) == 0:
        print(f'{scenario.scenario_id} does not exist in the submission file')

    plot_target_history(scenario, history,ax,False)

    i = 0
    #plot prediction
    if online_learning:
        # Get an iterator for the dictionary items
        iterator = iter(predictions_val.items())
        try:
            # Get the next prediction item
            next_key, _ = next(iterator)
        except StopIteration:
            # Handle the case when there are no more items
            next_key = -1
        exit_check = False
        for timestep_key, timestep_val in predictions_val.items():
            # Access the current prediction item
            try:
                # Get the next prediction item
                next_key, _ = next(iterator)
            except StopIteration:
                # Handle the case when there are no more items
                next_key = -1

            if next_key==-1 or (timestep_key >= timestep and timestep<next_key):
                for track_id, track_val in timestep_val.items():
                    i +=1
                    coordinates_array = track_val[0]
                    probabilities_array = track_val[1]

                    #plot prediction
                    _plot_polylines(coordinates_array, style='--',line_width=1, probabilities=probabilities_array)
                    exit_check = True
                
                if exit_check:
                    break
    else:
        for _, inner_value in predictions_val.items():
            i +=1
            coordinates_array = inner_value[0]
            probabilities_array = inner_value[1]

            #plot prediction
            _plot_polylines(coordinates_array, style='--',line_width=1, probabilities=probabilities_array)

    plt.show()

def plot_target_history(scenario: ArgoverseScenario, timestep: int=50, ax = None, show=True) -> None:
    if ax == None:
        _, ax = plt.subplots()
    
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep <= timestep]
        )
        actor_headings: NDArrayFloat = np.array(
            [object_state.heading for object_state in track.object_states if object_state.timestep <= timestep]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            # Get the current color
            if timestep <= 50:
                _plot_polylines([actor_trajectory], color=track_color, line_width=2)
            else:
                black = '#000000'
                # print('GT:',actor_trajectory[:51])
                _plot_polylines([actor_trajectory[:51]], color=track_color, line_width=2)
                _plot_polylines([actor_trajectory[50:]], color=black, line_width=2)


            if track.track_id == "AV":
                track_color = _AV_COLOR
            elif track.object_type in _STATIC_OBJECT_TYPES:
                continue

            # Plot bounding boxes for all vehicles and cyclists
            if track.object_type == ObjectType.VEHICLE:
                _plot_actor_bounding_box(
                    ax,
                    actor_trajectory[-1],
                    actor_headings[-1],
                    track_color,
                    (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
                )
            elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
                _plot_actor_bounding_box(
                    ax,
                    actor_trajectory[-1],
                    actor_headings[-1],
                    track_color,
                    (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
                )
            else:
                plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1, 1], "o", color=track_color, markersize=4)
    if show:
        plt.show()


def _plot_actor_tracks(ax: plt.Axes, scenario: ArgoverseScenario, timestep: int) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep <= timestep]
        )
        actor_headings: NDArrayFloat = np.array(
            [object_state.heading for object_state in track.object_states if object_state.timestep <= timestep]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            # Get the current color
            if timestep <= 50:
                _plot_polylines([actor_trajectory], color=track_color, line_width=2)
            else:
                black = '#000000'
                # print('GT:',actor_trajectory[:51])
                _plot_polylines([actor_trajectory[:51]], color=track_color, line_width=2)
                _plot_polylines([actor_trajectory[50:]], color=black, line_width=2)


        elif track.track_id == "AV":
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue

        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1, 1], "o", color=track_color, markersize=4)

    return track_bounds


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
    probabilities: Sequence[NDArrayFloat] = None,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    i = 0
    for polyline in polylines:
        if probabilities is not None:
            color = __probability_to_color(probabilities[i])  
        i+=1
        plt.plot(polyline[:, 0], polyline[:, 1], style, linewidth=line_width, color=color, alpha=alpha)


def _plot_polygons(polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r") -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(
    ax: plt.Axes, cur_location: NDArrayFloat, heading: float, color: str, bbox_size: Tuple[float, float]
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, angle=np.degrees(heading), color=color, zorder=_BOUNDING_BOX_ZORDER
    )
    ax.add_patch(vehicle_bounding_box)
