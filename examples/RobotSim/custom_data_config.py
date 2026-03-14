from gr00t.experiment.data_config import DataConfig
from gr00t.data.transform.base import Compose
from gr00t.data.transform.video import VideoTransform
from gr00t.data.transform.state_action import StateActionTransform


class RobotSimDataConfig(DataConfig):
    @classmethod
    def modality_config(cls):
        return {
            "video": {
                "ego_view": {
                    "original_key": "observation.images.ego_view",
                }
            },
            "state": {
                "state": {},
            },
            "action": {
                "action": {},
            },
            "annotation": {
                "human.action.task_description": {},
            },
        }

    @classmethod
    def transform(cls):
        return Compose(
            [
                VideoTransform(
                    image_keys=["ego_view"],
                ),
                StateActionTransform(
                    state_key="state",
                    action_key="action",
                ),
            ]
        )
