"""
manipulate the gz models' positions directly
"""

from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node as gzNode 


def set_entity_pose(entity, x, y, z, orientation_w=1.0,service_name="/world/warehouse/set_pose"):
    """
    Set the entity pose by calling the Gazebo service.

    Parameters:
    - entity: The name of the entity (e.g., x500_0, delivery_robot_0).
    - x, y, z: The position coordinates.
    - orientation_w: The w component of the quaternion orientation (default is 1.0).
    - service_name: The Gazebo service name for setting the pose.
    """
    node = gzNode()
    request = Pose()
    request.name = entity
    request.position.x = x
    request.position.y = y
    request.position.z = z
    request.orientation.x = 0.0
    request.orientation.y = 0.0
    request.orientation.z = 0.0
    request.orientation.w = orientation_w
    response = Boolean()
    timeout = 200  # Timeout in milliseconds

    result, response = node.request(service_name, request, Pose, Boolean, timeout)
    # print(f"Set {entity} position to ({x}, {y}, {z}), Result:", result, "\nResponse:", response.data)
