

from sqlalchemy import Column, Integer, String, LargeBinary
from .base import Base


class ImageDoubleFeature(Base):

    id = Column(Integer, primary_key=True)
    f_path = Column(String(length=255))
    feat_objective_id = Column(Integer)
    feat_scene_id = Column(Integer)
    feat_obj = Column(LargeBinary)
    feat_scene = Column(LargeBinary)

    __tablename__ = 'image_double_feature2'
