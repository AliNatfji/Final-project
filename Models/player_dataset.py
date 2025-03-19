from PIL import Image
import torch
import torchvision.transforms as transforms
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.base_dataset import _BaseDataset, _BaseDatasetItem
from Models.box import BoxInfo
from Utils.dataset import get_frame_img_path


class PlayerDataset(_BaseDataset):
    """
    Placeholder class for player-level classification dataset.
    Inherits from _Dataset.
    """

    def __init__(self, type: DatasetType):
        """
        Initializes the PlayerDataset by setting classification level to PLAYER.

        Args:
            type (DatasetType): Type of dataset (TRAIN, VAL, TEST).
        """
        super().__init__(type)

    def get_flatten(self) -> list[list['PlayerDatasetItem']]:
        dataset: list[list[PlayerDatasetItem]] = []
        for _, v in self._videos_annotations.items():
            for __, c in v.get_all_clips_annotations():
                items: dict[int, list[PlayerDatasetItem]] = {
                    i: [] for i in range(12)
                }
                for frame_ID, boxes in c.get_within_range_frame_boxes():
                    for box in boxes:
                        items[box.player_ID] += [PlayerDatasetItem(
                            video=v.video,
                            clip=c.clip,
                            frame=frame_ID,
                            img_path=get_frame_img_path(
                                v.video, c.clip, frame_ID),
                            box=box
                        )]
                for item in items.values():
                    dataset.append(item)
        return dataset

    # def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        items: list[PlayerDatasetItem] = self._flatten_dataset[index]

        player_imgs: list[torch.Tensor] = []
        for item in items:
            try:
                player_imgs += [Image.open(item.img_path).convert(
                    'RGB').crop(item.box.box)]
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at {item.img_path}")

        for i in range(len(player_imgs)):
            if self.has_bl_cf():
                player_imgs[i] = self.get_bl_cf().dataset.preprocess.get_transforms(
                    ClassificationLevel.PLAYER, self._type
                )(player_imgs[i])
            else:
                player_imgs[i] = transforms.ToTensor()(player_imgs[i])

        y_label = torch.Tensor(
            [self.get_cf().dataset.get_encoded_category(
                ClassificationLevel.PLAYER, item.box.category
            )]
        ).to(torch.long)
        return torch.stack(player_imgs), y_label[0]
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """ Retrieves an indexed sample from the dataset """

        items: list[PlayerDatasetItem] = self._flatten_dataset[index]

        player_imgs: list[torch.Tensor] = []
        y_label: torch.Tensor = torch.zeros(1, dtype=torch.long)  # Default label in case of empty items

        for item in items:
            try:
                img = Image.open(item.img_path).convert('RGB').crop(item.box.box)
                player_imgs.append(img)
            except FileNotFoundError:
                print(f"[WARNING] Image not found at {item.img_path}, skipping.")
                continue  # Skip missing images instead of raising an error

        if len(player_imgs) == 0:
            print(f"[ERROR] No valid player images found for index {index}. Returning placeholder tensor.")
            return torch.zeros((3, 50, 50)), torch.tensor(0)  # Placeholder

        # Apply transformations
        for i in range(len(player_imgs)):
            if self.has_bl_cf():
                player_imgs[i] = self.get_bl_cf().dataset.preprocess.get_transforms(
                    ClassificationLevel.PLAYER, self._type
                )(player_imgs[i])
            else:
                player_imgs[i] = transforms.ToTensor()(player_imgs[i])

        # Extract label from the first item
        y_label[0] = self.get_cf().dataset.get_encoded_category(
         ClassificationLevel.PLAYER, items[0].box.category
        )

        return torch.stack(player_imgs), y_label[0]

class PlayerDatasetItem(_BaseDatasetItem):
    def __init__(self, video: int, clip: int, frame: int, img_path: str, box: BoxInfo):
        super().__init__(video=video, clip=clip, frame=frame, img_path=img_path)
        self.box = box

    def to_dict(self):
        return dict([*super().to_dict().items(), ('box', self.box)])
