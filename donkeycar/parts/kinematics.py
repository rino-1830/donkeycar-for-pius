import logging
import math
import time
from typing import Tuple

"""運動学に関する部品を提供するモジュール."""

from donkeycar.utils import compare_to, sign, is_number_type, clamp

logger = logging.getLogger(__name__)


def limit_angle(angle: float):
    """角度を -\u03c0 から \u03c0 の範囲に正規化する."""
    return math.atan2(math.sin(angle), math.cos(angle))
    # twopi = math.pi * 2
    # while(angle > math.pi):
    #     angle -= twopi
    # while(angle < -math.pi):
    #     angle += twopi
    # return angle


class Pose2D:
    """2\u6b21\u5143\u7a7a\u9593\u306e\u59ff\u52e2\u3092\u8868\u3059\u7c21\u5358\u306a\u30c7\u30fc\u30bf\u69cb\u9020."""

    def __init__(self, x: float = 0.0, y: float = 0.0, angle: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.angle = angle


class Bicycle:
    """Ackermann\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u8eca\u4e21\u306e\u30d5\u30a9\u30ef\u30fc\u30c9\u30ad\u30cd\u30de\u30c6\u30a3\u30af\u30b9\u3092\u5b9f\u73fe\u3059\u308b\u81ea\u8ee2\u8eca\u30e2\u30c7\u30eb\u3002

    \u524d\u8eca\u8f2a\u306e\u4e2d\u70b9\u3092\u53c2\u7167\u70b9\u3068\u3057\u3066\u3001\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3068\u30aa\u30c9\u30e1\u30c8\u30ea\u306e\u8aad\u307f\u3092\u57fa\u306b\u3001\u524d\u9032\u8ddd\u96e2\u3068\u901f\u5ea6\u3001\u59ff\u52e2\u3001\u59ff\u52e2\u901f\u5ea6\u3092\u7b97\u51fa\u3059\u308b\u3002

    Args:
        wheel_base (float): \u524d\u8eca\u8f2a\u3068\u5f8c\u8eca\u8f2a\u9593\u306e\u8ddd\u96e2\u3002

    Note:
        \u3053\u306e\u30af\u30e9\u30b9\u306f\u524d\u8eca\u8f2a\u4e2d\u70b9\u3092\u53c2\u7167\u70b9\u3068\u3057\u3066\u4f7f\u7528\u3059\u308b\u3002
        https://thef1clan.com/2020/09/21/vehicle-dynamics-the-kinematic-bicycle-model/ \u3082\u53c2\u7167\u306e\u3053\u3068\u3002
    """
    def __init__(self, wheel_base:float, debug=False):
        self.wheel_base:float = wheel_base
        self.debug = debug
        self.timestamp:float = 0
        self.forward_distance:float = 0
        self.forward_velocity:float = 0
        self.steering_angle = None
        self.pose = Pose2D()
        self.pose_velocity = Pose2D()
        self.running:bool = True

    def run(
        self,
        forward_distance: float,
        steering_angle: float,
        timestamp: float | None = None,
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """\u30aa\u30c9\u30e1\u30c8\u30ea\u8a66\u8a3c\u7d50\u679c\u3068\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u304b\u3089\u59ff\u52e2\u3092\u66f4\u65b0\u3059\u308b\u3002

        Args:
            forward_distance (float): \u524d\u8eca\u8f2a\u4e2d\u70b9\u304c\u79fb\u52d5\u3057\u305f\u8ddd\u96e2\u3002
            steering_angle (float): \u524d\u8eca\u8f2a\u306e\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\uff08\u5de6\u304c\u6b63\uff09\u3002
            timestamp (float | None): \u8ddd\u96e2\u8a18\u9332\u306e\u6642\u523b\u3002\u6307\u5b9a\u3057\u306a\u3044\u5834\u5408\u306f\u73fe\u5728\u6642\u523b\u3092\u4f7f\u7528\u3059\u308b\u3002

        Returns:
            Tuple[float, float, float, float, float, float, float, float, float]:
                \u79fb\u52d5\u8ddd\u96e2\u3001\u901f\u5ea6\u3001x\u5ea7\u6a19\u3001y\u5ea7\u6a19\u3001\u65b9\u5411\u89d2\u3001x\u901f\u5ea6\u3001y\u901f\u5ea6\u3001\u89d2\u901f\u5ea6\u3001\u6642\u523b\u3002
        """
        if timestamp is None:
            timestamp = time.time()

        steering_angle = limit_angle(steering_angle)

        if self.running:
            if 0 == self.timestamp:
                self.forward_distance = 0
                self.forward_velocity=0
                self.steering_angle = steering_angle
                self.pose = Pose2D()
                self.pose_velocity = Pose2D()
                self.timestamp = timestamp
            elif timestamp > self.timestamp:
                #
                # \u524d\u56de\u306e\u5b9f\u884c\u304b\u3089\u306e\u5909\u5316
                #
                delta_time = timestamp - self.timestamp
                delta_distance = forward_distance - self.forward_distance
                delta_steering_angle = steering_angle - self.steering_angle

                #
                # \u65b0\u3057\u3044\u4f4d\u7f6e\u3068\u65b9\u5411
                # delta_time\u304c\u5c0f\u3055\u3044\u3068\u4eee\u5b9a\u3057\u3001\u52d5\u304d\u3092\u7dda\u5f62\u3068\u307f\u306a\u3059
                #
                # x, y, angle = update_bicycle_front_wheel_pose(
                #     self.pose,
                #     self.wheel_base,
                #     self.steering_angle + delta_steering_angle / 2,
                #     delta_distance)
                #
                # #
                # # \u901f\u5ea6\u306e\u66f4\u65b0
                # #
                # forward_velocity = delta_distance / delta_time
                # self.pose_velocity.angle = (angle - self.pose.angle) / delta_time
                # self.pose_velocity.x = (x - self.pose.x) / delta_time
                # self.pose_velocity.y = (y - self.pose.y) / delta_time
                # #
                # # \u59ff\u52e2\u306e\u66f4\u65b0
                # #
                # self.pose.x = x
                # self.pose.y = y
                # self.pose.angle = angle

                #
                # \u65b0\u3057\u3044\u901f\u5ea6\u5024
                #
                forward_velocity = delta_distance / delta_time
                self.pose_velocity.angle = bicycle_angular_velocity(self.wheel_base, forward_velocity, steering_angle)
                delta_angle = self.pose_velocity.angle * delta_time
                estimated_angle = limit_angle(self.pose.angle + delta_angle / 2)
                self.pose_velocity.x = forward_velocity * math.cos(estimated_angle)
                self.pose_velocity.y = forward_velocity * math.sin(estimated_angle)

                #
                # \u65b0\u3057\u3044\u59ff\u52e2
                #
                self.pose.x = self.pose.x + self.pose_velocity.x * delta_time
                self.pose.y = self.pose.y + self.pose_velocity.y * delta_time
                self.pose.angle = limit_angle(self.pose.angle + delta_angle)

                self.steering_angle = steering_angle

                #
                # オドメトリの更新
                #
                self.forward_distance = forward_distance
                self.forward_velocity = forward_velocity

                self.timestamp = timestamp

            result = (
                self.forward_distance,
                self.forward_velocity,
                self.pose.x, self.pose.y, self.pose.angle,
                self.pose_velocity.x, self.pose_velocity.y, self.pose_velocity.angle,
                self.timestamp
            )
            if self.debug:
                logger.info(result)
            return result

        return 0, 0, 0, 0, 0, 0, 0, 0, self.timestamp

    def shutdown(self):
        self.running = False


class InverseBicycle:
    """
    Bicycle inverse kinematics for a car-like vehicle (Ackerman steering)
    takes the forward velocity and the angular velocity in radians/second
    and converts these to:
    - forward velocity (pass through),
    - steering angle in radians
    @param wheel_base: distance between the front and back wheels

    NOTE: this version uses the point midway between the rear wheels
          as the point of reference.
    see https://thef1clan.com/2020/09/21/vehicle-dynamics-the-kinematic-bicycle-model/
    """
    def __init__(self, wheel_base:float, debug=False):
        self.wheel_base:float = wheel_base
        self.debug = debug
        self.timestamp:float = 0

    def run(self, forward_velocity:float, angular_velocity:float, timestamp:float=None) -> Tuple[float, float, float]:
        """
        @param forward_velocity:float in meters per second
        @param angular_velocity:float in radians per second
        @return tuple
                - forward_velocity:float in meters per second (basically a pass through)
                - steering_angle:float in radians
                - timestamp:float
        """
        if timestamp is None:
            timestamp = time.time()

        """
        derivation from bicycle model:
        angular_velocity = forward_velocity * math.tan(steering_angle) / self.wheel_base
        math.tan(steering_angle) = angular_velocity * self.wheel_base / forward_velocity
        steering_angle = math.atan(angular_velocity * self.wheel_base / forward_velocity)
        """
        steering_angle = bicycle_steering_angle(self.wheel_base, forward_velocity, angular_velocity)        
        self.timestamp = timestamp

        return forward_velocity, steering_angle, timestamp


def update_bicycle_front_wheel_pose(front_wheel, wheel_base, steering_angle, distance):
    """
    Calculates the ending position of the front wheel of a bicycle kinematics model.
    This is expected to be called at a high rate such that we can model the
    the travel as a line rather than an arc.

    Arguments:
    front_wheel -- starting pose at front wheel as tuple of (x, y, angle) where
                x -- initial x-coordinate of the front wheel (float)
                y -- initial y-coordinate of the front wheel (float)
                angle -- initial orientation of the vehicle along it's wheel base (in radians) (float)
    wheel_base -- length of the wheel base (float)
    steering_angle -- steering angle (in radians) (float)
    distance -- distance travelled by the vehicle (float)

    Returns:
    A tuple (x_f, y_f, theta_f) representing the ending position and orientation of the front wheel.
    x_f -- ending x-coordinate of the front wheel (float)
    y_f -- ending y-coordinate of the front wheel (float)
    theta_f -- ending orientation of the vehicle (in radians) (float)
    """
    if distance == 0:
        return front_wheel

    if steering_angle == 0:
        x = front_wheel.x + distance * math.cos(front_wheel.angle)
        y = front_wheel.y + distance * math.sin(front_wheel.angle)
        theta = front_wheel.angle
    else:
        theta = limit_angle(front_wheel.angle + math.tan(steering_angle) * distance / wheel_base)
        x = front_wheel.x + distance * math.cos(theta)
        y = front_wheel.y + distance * math.sin(theta)
    return x, y, theta


def bicycle_steering_angle(wheel_base:float, forward_velocity:float, angular_velocity:float) -> float:
    """
    Calculate bicycle steering for the vehicle from the angular velocity.
    For car-like vehicles, calculate the angular velocity using 
    the bicycle model and the measured max forward velocity and max steering angle.
    """
    #
    # derivation from bicycle model:
    # angular_velocity = forward_velocity * math.tan(steering_angle) / self.wheel_base
    # math.tan(steering_angle) = angular_velocity * self.wheel_base / forward_velocity
    # steering_angle = math.atan(angular_velocity * self.wheel_base / forward_velocity)
    #
    # return math.atan(angular_velocity * wheel_base / forward_velocity)
    return limit_angle(math.asin(angular_velocity * wheel_base / forward_velocity))


def bicycle_angular_velocity(wheel_base:float, forward_velocity:float, steering_angle:float) -> float:
    """
    Calculate angular velocity for the vehicle from the bicycle steering angle.
    For car-like vehicles, calculate the angular velocity using 
    the bicycle model and the measured max forward velocity and max steering angle.
    """
    #
    # for car-like (bicycle model) vehicle, for the back axle:
    # angular_velocity = forward_velocity * math.tan(steering_angle) /  wheel_base if velocity is from rear wheels
    # angular_velocity = forward_velocity * math.tan(steering_angle) /  wheel_base if velocity is from front wheels
    #
    # return forward_velocity * math.tan(steering_angle) / wheel_base # \u5f8c\u8eca\u8f2a\u7528\u306e\u901f\u5ea6
    return forward_velocity * math.sin(steering_angle) / wheel_base  # \u524d\u8eca\u8f2a\u306e\u901f\u5ea6


class BicycleNormalizeAngularVelocity:
    """\u8eca\u5f0f\u30d0\u30a4\u30b7\u30af\u30eb\u30e2\u30c7\u30eb\u306e\u89d2\u901f\u5ea6\u3092\u5b9f\u6570\u304b\u3089\u6b63\u898f\u5316\u5024\u3078\u7b97\u51fa\u3059\u308b\u30d1\u30fc\u30c4\u3002

    Args:
        wheel_base (float): \u8eca\u8f2a\u57fa\u7dda\u306e\u9577\u3055\u3002
        max_forward_velocity (float): \u6e2c\u5b9a\u3055\u308c\u305f\u6700\u9ad8\u901f\u5ea6\u3002
        max_steering_angle (float): \u6e2c\u5b9a\u3055\u308c\u305f\u6700\u5927\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3002
    """
    def __init__(self, wheel_base:float, max_forward_velocity:float, max_steering_angle:float) -> None:
        self.max_angular_velocity = bicycle_angular_velocity(wheel_base, max_forward_velocity, max_steering_angle)

    def run(self, angular_velocity: float) -> float:
        """\u89d2\u901f\u5ea6\u3092\u7bc4\u56f2 -1 \u304b\u3089 1 \u306b\u5909\u63db\u3059\u308b."""
        return angular_velocity / self.max_angular_velocity


class BicycleUnnormalizeAngularVelocity:
    """\u6b63\u898f\u5316\u3055\u308c\u305f\u89d2\u901f\u5ea6\u3092\u5b9f\u5024\u306e\u89d2\u901f\u5ea6\u306b\u5909\u63db\u3059\u308b\u30d1\u30fc\u30c4\u3002"""
    def __init__(self, wheel_base:float, max_forward_velocity:float, max_steering_angle:float) -> None:
        self.max_angular_velocity = bicycle_angular_velocity(wheel_base, max_forward_velocity, max_steering_angle)

    def run(self, normalized_angular_velocity: float) -> float:
        """\u6b63\u898f\u5316\u3055\u308c\u305f\u5024\u304b\u3089\u89d2\u901f\u5ea6\u3092\u5f97\u308b."""
        if abs(normalized_angular_velocity) > 1:
            logger.error("\u8b66\u544a: normalized_angular_velocity \u306f -1 \u304b\u3089 1 \u306e\u9593\u3067\u3042\u308b\u3079\u304d")
        return normalized_angular_velocity * self.max_angular_velocity


class Unicycle:
    """\u30c7\u30d5\u30a1\u30ec\u30f3\u30b7\u30e3\u30eb\u30c9\u30e9\u30a4\u30d6\u8eca\u4e21\u306e\u9802\u70b9\u30ad\u30cd\u30de\u30c6\u30a3\u30af\u30b9\u3002

    \u5de6\u8eca\u8f2a\u3068\u53f3\u8eca\u8f2a\u306e\u30aa\u30c9\u30e1\u30fc\u30bf\u3092\u5165\u529b\u3068\u3057\u3001\u524d\u9032\u8ddd\u96e2\u3084\u901f\u5ea6\u3001\u59ff\u52e2\u3092\u7b97\u51fa\u3059\u308b\u3002

    Args:
        axle_length (float): \u4e21\u8eca\u8f2a\u9593\u306e\u8ddd\u96e2\u3002

    Note:
        http://faculty.salina.k-state.edu/tim/robotics_sg/Control/kinematics/unicycle.html を参照。
    """
    def __init__(self, axle_length:float, debug=False):
        self.axle_length:float = axle_length
        self.debug = debug
        self.timestamp:float = 0
        self.left_distance:float = 0
        self.right_distance:float = 0
        self.velocity:float = 0
        self.pose = Pose2D()
        self.pose_velocity = Pose2D()
        self.running:bool = True

    def run(
        self,
        left_distance: float,
        right_distance: float,
        timestamp: float | None = None,
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """\u30aa\u30c9\u30e1\u30c8\u30ea\u7d50\u679c\u304b\u3089\u59ff\u52e2\u3092\u7b97\u51fa\u3059\u3002

        Args:
            left_distance (float): \u5de6\u8eca\u8f2a\u304c\u9032\u3093\u3060\u8ddd\u96e2\u3002
            right_distance (float): \u53f3\u8eca\u8f2a\u304c\u9032\u3093\u3060\u8ddd\u96e2\u3002
            timestamp (float | None): \u8ddd\u96e2\u8aad\u307f\u53d6\u308a\u6642\u523b\u3002None \u306a\u3089\u73fe\u5728\u6642\u523b\u3002

        Returns:
            Tuple[float, float, float, float, float, float, float, float, float]:
                \u79fb\u52d5\u8ddd\u96e2\u3001\u901f\u5ea6\u3001x\u5ea7\u6a19\u3001y\u5ea7\u6a19\u3001\u65b9\u5411\u89d2\u3001x\u901f\u5ea6\u3001y\u901f\u5ea6\u3001\u89d2\u901f\u5ea6\u3001\u6642\u523b\u3002
        """
        if timestamp is None:
            timestamp = time.time()

        if self.running:
            if 0 == self.timestamp:
                self.timestamp = timestamp
                self.left_distance = left_distance
                self.right_distance = right_distance
                self.velocity=0
                self.pose = Pose2D()
                self.pose_velocity = Pose2D()
                self.timestamp = timestamp
            elif timestamp > self.timestamp:
                #
                # \u524d\u56de\u306e\u5b9f\u884c\u304b\u3089\u306e\u5909\u5316
                #
                delta_left_distance = left_distance - self.left_distance
                delta_right_distance = right_distance - self.right_distance
                delta_distance = (delta_left_distance + delta_right_distance) / 2
                delta_angle = (delta_right_distance - delta_left_distance) / self.axle_length
                delta_time = timestamp - self.timestamp

                forward_velocity = delta_distance / delta_time
                angle_velocity = delta_angle / delta_time

                #
                # \u65b0\u3057\u3044\u4f4d\u7f6e\u3068\u65b9\u5411
                #
                estimated_angle = limit_angle(self.pose.angle + delta_angle / 2)
                x = self.pose.x + delta_distance * math.cos(estimated_angle)
                y = self.pose.y + delta_distance * math.sin(estimated_angle)
                angle = limit_angle(self.pose.angle + delta_angle)

                #
                # \u65b0\u3057\u3044\u901f\u5ea6\u5024
                #
                self.pose_velocity.x = (x - self.pose.x) / delta_time
                self.pose_velocity.y = (y - self.pose.y) / delta_time
                self.pose_velocity.angle = angle_velocity

                #
                # \u59ff\u52e2\u3092\u66f4\u65b0
                #
                self.pose.x = x
                self.pose.y = y
                self.pose.angle = angle

                #
                # \u30aa\u30c9\u30e1\u30c8\u30ea\u306e\u66f4\u65b0
                #
                self.left_distance = left_distance
                self.right_distance = right_distance
                self.velocity = forward_velocity

                self.timestamp = timestamp

            return (
                (self.left_distance + self.right_distance) / 2,
                self.velocity,
                self.pose.x, self.pose.y, self.pose.angle,
                self.pose_velocity.x, self.pose_velocity.y, self.pose_velocity.angle,
                self.timestamp
            )

        return 0, 0, 0, 0, 0, 0, 0, 0, self.timestamp

    def shutdown(self):
        self.running = False


class InverseUnicycle:
    """\u30c7\u30d5\u30a1\u30ec\u30f3\u30b7\u30e3\u30eb\u306e\u9006\u7b97\u904b\u52d5\u5b66\u3092\u5b9f\u88c5\u3059\u308b\u30d1\u30fc\u30c4\u3002

    \u524d\u9032\u901f\u5ea6\u3068\u89d2\u901f\u5ea6\u304b\u3089\u3001\u5de6\u53f3\u8eca\u8f2a\u306e\u7dda\u5f62\u901f\u5ea6\u3092\u7b97\u51fa\u3059\u3002
    """
    def __init__(self, axle_length:float, wheel_radius:float, min_speed:float, max_speed:float, steering_zero:float=0.01, debug=False):
        self.axle_length:float = axle_length
        self.wheel_radius:float = wheel_radius
        self.min_speed:float = min_speed
        self.max_speed:float = max_speed
        self.steering_zero:float = steering_zero
        self.timestamp = 0
        self.debug = debug

        self.wheel_diameter = 2 * wheel_radius
        self.wheel_circumference = math.pi * self.wheel_diameter

    def run(
        self,
        forward_velocity: float,
        angular_velocity: float,
        timestamp: float | None = None,
    ) -> Tuple[float, float, float]:
        """\u8ee2\u56de\u901f\u5ea6\u3068\u524d\u9032\u901f\u5ea6\u304b\u3089\u8eca\u8f2a\u901f\u5ea6\u3092\u7b97\u51fa\u3059\u3002

        Args:
            forward_velocity (float): \u524d\u9032\u901f\u5ea6\uff08m/s\uff09。
            angular_velocity (float): \u89d2\u901f\u5ea6\uff08rad/s\uff09。
            timestamp (float | None): \u30c7\u30fc\u30bf\u53d6\u5f97\u6642\u523b\u3002None \u306a\u3089\u73fe\u5728\u6642\u523b\u3002

        Returns:
            Tuple[float, float, float]: \u5de6\u8eca\u8f2a\u901f\u5ea6\u3001\u53f3\u8eca\u8f2a\u901f\u5ea6\u3001\u6642\u523b\u3002
        """
        if timestamp is None:
            timestamp = time.time()

        left_linear_speed = forward_velocity - angular_velocity * self.axle_length / 2
        right_linear_speed = forward_velocity + angular_velocity * self.axle_length / 2

        self.timestamp = timestamp

        # \u5de6\u53f3\u8eca\u8f2a\u306e\u7dda\u5f62\u901f\u5ea6\u3068\u6642\u523b
        return left_linear_speed, right_linear_speed, timestamp

    def shutdown(self):
        pass


def unicycle_angular_velocity(wheel_radius:float, axle_length:float, left_velocity:float, right_velocity:float) -> float:
    """
    Calculate angular velocity for the unicycle vehicle.
    For differential drive, calculate angular velocity 
    using the unicycle model and linear wheel velocities. 
    """
    #
    # angular_velocity = wheel_radius / axle_length * (right_rotational_velocity - left_rotational_velocity)
    # という関係からの計算
    # ホイール転速はラジアン/秒である
    #
    right_rotational_velocity = wheel_rotational_velocity(wheel_radius, right_velocity)
    left_rotational_velocity = wheel_rotational_velocity(wheel_radius, left_velocity)
    return wheel_radius / axle_length * (right_rotational_velocity - left_rotational_velocity)


def unicycle_max_angular_velocity(wheel_radius:float, axle_length:float, max_forward_velocity:float) -> float:
    """
    Calculate maximum angular velocity for the vehicle, so we can convert between
    normalized and unnormalized forms of the angular velocity.
    For differential drive, calculate maximum angular velocity 
    using the unicycle model and assuming one 
    one wheel is stopped and one wheel is at max velocity.
    """
    #
    # angular_velocity = wheel_radius / axle_length * (right_rotational_velocity - left_rotational_velocity)
    # ホイール転速はラジアン/秒である
    # 右車輪を最大速度で回し左車輪を停止させると
    # max_angular_velocity = wheel_radius / axle_length * max_forward_velocity
    #
    return unicycle_angular_velocity(wheel_radius, axle_length, 0, max_forward_velocity)


class UnicycleNormalizeAngularVelocity:
    """\u8eca\u8f2a\u5229\u7528\u578b\u8eca\u4e21\u306e\u89d2\u901f\u5ea6\u3092\u6b63\u898f\u5316\u3059\u308b\u30d1\u30fc\u30c4\u3002"""
    def __init__(self, wheel_radius:float, axle_length:float, max_forward_velocity:float) -> None:
        self.max_angular_velocity = unicycle_max_angular_velocity(wheel_radius, axle_length, max_forward_velocity)

    def run(self, angular_velocity: float) -> float:
        """\u89d2\u901f\u5ea6\u3092 -1 \u304b\u3089 1 \u306b\u5909\u63db\u3059\u308b."""
        return angular_velocity / self.max_angular_velocity


class UnicycleUnnormalizeAngularVelocity:
    """\u6b63\u898f\u5316\u89d2\u901f\u5ea6\u3092\u5b9f\u969b\u306e\u89d2\u901f\u5ea6\u306b\u5909\u63db\u3059\u308b\u30d1\u30fc\u30c4\u3002"""
    def __init__(self, wheel_radius:float, axle_length:float, max_forward_velocity:float) -> None:
        self.max_angular_velocity = unicycle_max_angular_velocity(wheel_radius, axle_length, max_forward_velocity)

    def run(self, normalized_angular_velocity: float) -> float:
        """\u6b63\u898f\u5316\u5024\u304b\u3089\u89d2\u901f\u5ea6\u3092\u8a08\u7b97\u3059\u308b."""
        if abs(normalized_angular_velocity) > 1:
            logger.error("\u8b66\u544a: normalized_angular_velocity \u306f -1 \u304b\u3089 1 \u306e\u9593\u3067\u3042\u308b\u3079\u304d")
        return normalized_angular_velocity * self.max_angular_velocity


class NormalizeSteeringAngle:
    """\u5b9f\u969b\u306e\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3092\u6b63\u898f\u5316\u5024\u3078\u5909\u63db\u3059\u308b\u30d1\u30fc\u30c4\u3002"""
    def __init__(self, max_steering_angle:float, steering_zero:float=0.0) -> None:
        """\u6e2c\u5b9a\u3055\u308c\u305f\u6700\u5927\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3068\u3001\u30bc\u30ed\u5224\u5b9a\u5024\u3092\u8a2d\u5b9a\u3059\u308b\u3002"""
        self.max_steering_angle = max_steering_angle
        self.steering_zero = steering_zero
    
    def run(self, steering_angle) -> float:
        """\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3092\u6b63\u898f\u5316\u3057\u305f\u5024\u306b\u5909\u63db\u3059\u308b."""
        if not is_number_type(steering_angle):
            logger.error("\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u306f\u6570\u5024\u3067\u306a\u3051\u308c\u3070\u306a\u308a\u307e\u305b\u3093")
            return 0

        steering = steering_angle / self.max_steering_angle
        if abs(steering) <= self.steering_zero:
            return 0
        return -steering # \u5b9f\u5728\u306e\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u304c\u6b63\u306e\u5834\u5408\u3001\u30ce\u30fc\u30de\u30e9\u30a4\u30ba\u5024\u306f\u8ca0\u306b\u306a\u308b

    def shutdown(self):
        pass


class UnnormalizeSteeringAngle:
    """\u6b63\u898f\u5316\u3055\u308c\u305f\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u3092\u5b9f\u969b\u306e\u89d2\u5ea6\u306b\u5909\u63db\u3059\u308b\u30d1\u30fc\u30c4\u3002"""
    def __init__(self, max_steering_angle:float, steering_zero:float=0.0) -> None:
        """\u6e2c\u5b9a\u3055\u308c\u305f\u6700\u5927\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3068\u30bc\u30ed\u5224\u5b9a\u5024\u3092\u8a2d\u5b9a\u3059\u308b\u3002"""
        self.max_steering_angle = max_steering_angle
        self.steering_zero = steering_zero
    
    def run(self, steering) -> float:
        """\u6b63\u898f\u5316\u5024\u3092\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u306b\u5909\u63db\u3059\u308b."""
        if not is_number_type(steering):
            logger.error("\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u306f\u6570\u5024\u3067\u306a\u3051\u308c\u3070\u306a\u308a\u307e\u305b\u3093")
            return 0

        if steering > 1 or steering < -1:
            logger.warn(f"steering = {steering}, \u3067\u3059\u304c 1\u304b\u3089 -1 \u306e\u7bc4\u56f2\u3067\u3042\u308b\u3079\u304d")

        steering = clamp(steering, -1, 1)
        
        s = sign(steering)
        steering = abs(steering)
        if steering <= self.steering_zero:
            return 0.0

        return self.max_steering_angle * steering * -s

    def shutdown(self):
        pass


def wheel_rotational_velocity(wheel_radius:float, speed:float) -> float:
    """\u63a8\u9032\u901f\u5ea6\u3092\u30db\u30a4\u30fc\u30eb\u306e\u8ee2\u901f\u306b\u5909\u63db\u3059\u308b\u3002

    Args:
        wheel_radius (float): \u8eca\u8f2a\u306e\u534a\u5f84\u3002
        speed (float): \u540c\u3058\u5358\u4f4d\u306e\u901f\u5ea6\u5024\u3002

    Returns:
        float: \u8ee2\u79d2\u5358\u4f4d\u306e\u30db\u30a4\u30fc\u30eb\u8ee2\u901f\u3002
    """
    return speed / wheel_radius


def differential_steering(throttle: float, steering: float, steering_zero: float = 0) -> Tuple[float, float]:
        """\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u89d2\u3068\u30b9\u30ed\u30c3\u30c8\u30eb\u3092\u5de6\u53f3\u8eca\u8f2a\u306e\u901f\u5ea6\u306b\u5909\u63db\u3059\u308b\u3002

        \u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u5024\u306b\u5bfe\u5fdc\u3057\u3066\u7247\u5074\u306e\u8eca\u8f2a\u3092\u904b\u52d5\u7de9\u6162\u3055\u305b\u308b\u306e\u3067\u3001\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u7121\u3057\u4ee5\u5916\u306f\u76f8\u5bfe\u7684\u306b\u4f4e\u901f\u3068\u306a\u308b\u3002
        \u3053\u308c\u306f\u53cd\u5f8c\u30ad\u30cd\u30de\u30c6\u30a3\u30af\u30b9\u3067\u306f\u306a\u304f\u3001\u30e6\u30fc\u30b6\u64cd\u4f5c\u6642\u306e\u30c8\u30e9\u30c3\u30d7\u5236\u5fa1\u306b\u9069\u3057\u3066\u3044\u308b\u3002

        Args:
            throttle (float): \u5b9f\u901f\u5ea6\u307e\u305f\u306f\u30b9\u30ed\u30c3\u30c8\u30eb\uff08-1\u304b\u30891\uff09。
            steering (float): -1\u304c\u5de6\u30011\u304c\u53f3\u3092\u8868\u3059\u6b63\u898f\u5316\u5024。
            steering_zero (float): \u3053\u306e\u5024\u4ee5\u4e0b\u306f\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u306a\u3057\u3068\u307f\u306a\u3059。
        """
        if not is_number_type(throttle):
            logger.error("throttle \u306f\u6570\u5024\u3067\u3042\u308b\u3079\u304d")
            return 0, 0
        if throttle > 1 or throttle < -1:
            logger.warn(f"throttle = {throttle}, \u3067\u3059\u304c 1\u304b\u3089 -1 \u306e\u7bc4\u56f2\u3067\u3042\u308b\u3079\u304d")
        throttle = clamp(throttle, -1, 1)

        if not is_number_type(steering):
            logger.error("\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u306f\u6570\u5024\u3067\u306a\u3051\u308c\u3070\u306a\u308a\u307e\u305b\u3093")
            return 0, 0
        if steering > 1 or steering < -1:
            logger.warn(f"steering = {steering}, \u3067\u3059\u304c 1\u304b\u3089 -1 \u306e\u7bc4\u56f2\u3067\u3042\u308b\u3079\u304d")
        steering = clamp(steering, -1, 1)

        left_throttle = throttle
        right_throttle = throttle
 
        if steering < -steering_zero:
            left_throttle *= (1.0 + steering)
        elif steering > steering_zero:
            right_throttle *= (1.0 - steering)

        return left_throttle, right_throttle        


class TwoWheelSteeringThrottle:
    """\u30c7\u30d5\u30a1\u30ec\u30f3\u30b7\u30e3\u30eb\u8eca\u4e21\u3067\u30b9\u30ed\u30c3\u30c8\u30eb\u3068\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u3092\u500b\u5225\u306e\u8eca\u8f2a\u901f\u5ea6\u306b\u5909\u63db\u3059\u308b\u30d1\u30fc\u30c4\u3002

    Args:
        steering_zero (float): \u3053\u306e\u5024\u4ee5\u4e0b\u306e\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u306f0\u3068\u307f\u306a\u3059\u3002
    """
    def __init__(self, steering_zero: float = 0.01) -> None:
        if not is_number_type(steering_zero):
            raise ValueError("steering_zero \u306f\u6570\u5024\u3067\u3042\u308b\u3079\u304d")
        if steering_zero > 1 or steering_zero < 0:
            raise ValueError(f"steering_zero {steering_zero} \u306f 1 \u304b\u3089 0 \u306e\u7bc4\u56f2\u3067\u3042\u308b\u3079\u304d")
        self.steering_zero = steering_zero

    def run(self, throttle, steering):
        """\u30b9\u30ed\u30c3\u30c8\u30eb\u3068\u30b9\u30c6\u30a2\u30ea\u30f3\u30b0\u3092\u500b\u5225\u306e\u8eca\u8f2a\u901f\u5ea6\u306b\u5909\u63db\u3059\u308b."""
        return differential_steering(throttle, steering, self.steering_zero)
 
    def shutdown(self):
        pass
