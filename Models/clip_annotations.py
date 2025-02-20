from collections import defaultdict
from Models.box import BoxInfo
from Models.config_mixin import _ConfigMixin
from Utils.dataset import get_frame_img_path, get_players_box_annot_path
from Utils.visualization import add_box, show_clip, show_image
import cv2


class ClipAnnotations(_ConfigMixin):
    def __init__(self, video: int, clip: int, category: str):
        super().__init__()
        self.video: int = video
        self.clip: int = clip
        self.__category: str = category
        self.__boxes = defaultdict(list[BoxInfo])
        _annot_path = get_players_box_annot_path(video, clip)

        with open(_annot_path, 'r') as file:
            for line in file:
                self.__add_box(BoxInfo(line))

    def __add_box(self, box: BoxInfo):
        if box.player_ID > 11:
            return
        self.__boxes[box.frame_ID].append(box)

    def get_frame_boxes(self, frame_ID: int) -> list[BoxInfo]:
        return self.__boxes[frame_ID]

    def get_all_frames_boxes(self):
        return self.__boxes.items()

    def get_within_range_frame_boxes(self):
        past, post = (0, 0) if not self.has_bl_cf() else \
            (
            self.get_bl_cf().dataset.past_frames_count,
            self.get_bl_cf().dataset.post_frames_count
        )

        frames = sorted(self.__boxes.keys())
        target_idx = frames.index(self.clip)
        filtered: dict[int, list[BoxInfo]] = {frame_ID: self.__boxes[frame_ID]
                                              for frame_ID in frames[(target_idx-past):(target_idx+post+1)]}
        return filtered.items()

    def get_category(self):
        return self.__category

    def show_frame_with_boxes(self, frame_ID: int, figsize=(80, 20)):
        image = self.__load_img_and_add_boxes(frame_ID)
        show_image(
            image, f'Video {self.video} - Frame {frame_ID}', figsize=figsize)

    def show_clip_with_boxes(self):
        images = []
        for frame_ID, boxes_info in self.get_all_frames_boxes():
            image = self.__load_img_and_add_boxes(frame_ID)
            images.append(image)

        show_clip(images)

    def __load_img_and_add_boxes(self, frame_ID):
        img_path = get_frame_img_path(self.video, self.clip, frame_ID)
        image = cv2.imread(img_path)

        for box_info in self.get_frame_boxes(frame_ID):
            add_box(image, box_info)

        return image
