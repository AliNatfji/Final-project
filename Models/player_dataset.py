from PIL import Image, ImageFile
import torch
import torchvision.transforms as transforms
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.base_dataset import _BaseDataset, _BaseDatasetItem
from Models.box import BoxInfo
from Utils.dataset import get_frame_img_path

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Prevent crashes on truncated images

class PlayerDataset(_BaseDataset):
    def __init__(self, type: DatasetType):
        super().__init__(type)
        self._flatten_dataset = self.get_flatten()

    def get_flatten(self) -> list[list['PlayerDatasetItem']]:
        dataset = []
        for _, video_annotation in self._videos_annotations.items():
            for __, clip in video_annotation.get_all_clips_annotations():
                players_dict = {i: [] for i in range(12)}
                for frame_id, boxes in clip.get_within_range_frame_boxes():
                    for box in boxes:
                        item = PlayerDatasetItem(
                            video=video_annotation.video,
                            clip=clip.clip,
                            frame=frame_id,
                            img_path=get_frame_img_path(video_annotation.video, clip.clip, frame_id),
                            box=box
                        )
                        players_dict[box.player_ID].append(item)
                dataset.extend(players_dict.values())
        return dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        items = self._flatten_dataset[index]
        images = []

        for item in items:
            try:
                img = Image.open(item.img_path).convert("RGB").crop(item.box.box)
                images.append(img)
            except Exception as e:
                print(f"[WARNING] Skipping image at {item.img_path}: {e}")

        if not images:
            print(f"[ERROR] No valid player images found at index {index}")
            dummy = Image.new("RGB", (224, 224))
            tensor = self.get_bl_cf().dataset.preprocess.get_transforms(
                ClassificationLevel.PLAYER, self._type
            )(dummy)
            return tensor.unsqueeze(0), torch.tensor(0)

        for i in range(len(images)):
            images[i] = self.get_bl_cf().dataset.preprocess.get_transforms(
                ClassificationLevel.PLAYER, self._type
            )(images[i])

        label = self.get_cf().dataset.get_encoded_category(
            ClassificationLevel.PLAYER, items[0].box.category
        )

        return torch.stack(images), torch.tensor(label)

    def __len__(self) -> int:
        return len(self._flatten_dataset)


class PlayerDatasetItem(_BaseDatasetItem):
    def __init__(self, video: int, clip: int, frame: int, img_path: str, box: BoxInfo):
        super().__init__(video=video, clip=clip, frame=frame, img_path=img_path)
        self.box = box

    def to_dict(self):
        return dict([*super().to_dict().items(), ('box', self.box)])