import rospy
from std_msgs.msg import String, Int32, Float32

"""sudo apt-get install python3-catkin-pkg

ROS と Python3 の問題:
https://discourse.ros.org/t/should-we-warn-new-users-about-difficulties-with-python-3-and-alternative-python-interpreters/3874/3
"""

class RosPubisher(object):
    """データストリームへパブリッシュする ROS ノード。"""

    def __init__(self, node_name, channel_name, stream_type=String, anonymous=True):
        """オブジェクトを初期化する。

        Args:
            node_name: ノード名。
            channel_name: チャネル名。
            stream_type: パブリッシュするメッセージ型。
            anonymous: `True` の場合、匿名でノードを起動する。
        """
        self.data = ""
        self.pub = rospy.Publisher(channel_name, stream_type)
        rospy.init_node(node_name, anonymous=anonymous)

    def run(self, data):
        """データが変化した時のみパブリッシュする。

        Args:
            data: 送信するデータ。
        """
        if data != self.data and not rospy.is_shutdown():
            self.data = data
            self.pub.publish(data)
    

class RosSubscriber(object):
    """データストリームをサブスクライブする ROS ノード。"""

    def __init__(self, node_name, channel_name, stream_type=String, anonymous=True):
        """オブジェクトを初期化する。

        Args:
            node_name: ノード名。
            channel_name: チャネル名。
            stream_type: 購読するメッセージ型。
            anonymous: `True` の場合、匿名でノードを起動する。
        """
        self.data = ""
        rospy.init_node(node_name, anonymous=anonymous)
        self.pub = rospy.Subscriber(channel_name, stream_type, self.on_data_recv)

    def on_data_recv(self, data):
        """受信したデータを保存する。"""
        self.data = data.data

    def run(self):
        """最新のデータを返す。"""
        return self.data

