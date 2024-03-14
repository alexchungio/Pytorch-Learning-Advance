"""
install rosbag: pip install --extra-index-url https://rospypi.github.io/simple/ rosbag
install cv_bridge: https://codeload.github.com/ros-perception/vision_opencv/zip/refs/heads/noetic
install sensor_msgs: pip install --extra-index-url https://rospypi.github.io/simple/ sensor_msgs
"""
import os
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tqdm


class ExtractBag(object):

    def __init__(self, bag_path, topic_names, save_path):
        self.bag_path = bag_path
        self.topic_names = topic_names
        self.save_path = save_path

    def extract(self):
        bag = rosbag.Bag(self.bag_path, 'r')  # read bag
        bridge = CvBridge()  # convert msgs to image

        for topic_name in self.topic_names:
            topic_path = self.save_path + topic_name
            os.makedirs(topic_path, exist_ok=True)

            topic_data = bag.read_messages(topic_name)
            pbar = tqdm.tqdm(topic_data)
            index = 0
            for topic, msg, t in pbar:
                timestamp = msg.header.stamp.to_sec()
                pbar.set_description(f'Processing extract image id: {timestamp}')
                cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
                second, millisecond = str(timestamp).split('.')
                if len(millisecond) < 6:
                    continue
                else:
                    base_name = second + '_' + millisecond
                img_path = os.path.join(topic_path, base_name + '.jpg')
                if 'fs' in topic or 'sign' in topic:
                    img_path = img_path.replace('.jpg', '.png')
                cv2.imwrite(img_path, cv_image, params=[cv2.IMWRITE_AVIF_QUALITY, 100])
                index = 1
            print(f'extract {index} images from {topic_name}')

    def sync(self):
        cur_list = None
        # get intersection
        for topic_name in self.topic_names:
            topic_dir = self.bag_path.replace('.bag', '') + topic_name
            if cur_list is None:
                cur_list = [name.split('.')[0] for name in os.listdir(topic_dir)]
                continue
            else:
                next_list = [name.split('.')[0] for name in os.listdir(topic_dir)]
                cur_list = list(set(cur_list).intersection(set(next_list)))
        # remove non-intersection
        for topic_name in self.topic_names:
            topic_dir = self.bag_path.replace('.bag', '') + topic_name
            for file_name in os.listdir(topic_dir):
                if file_name.split('.')[0] not in cur_list:
                    os.remove(os.path.join(topic_dir, file_name))
            print(f'Remain {len(os.listdir(topic_dir))} image in {topic_name}')


if __name__ == "__main__":
    topic_names = ["/aroundview_output_fs", "/bev_image_data"]
    bag_path = "/Users/alex/Documents/ros_bag/bag_0313/new/2024-03-13-20-10-50.bag"
    save_path = os.path.join(os.path.dirname(bag_path), os.path.basename(bag_path).split('.')[0])
    extractor = ExtractBag(bag_path, topic_names, save_path)
    extractor.extract()
    extractor.sync()