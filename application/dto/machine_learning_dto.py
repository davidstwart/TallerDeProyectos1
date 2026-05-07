from pydantic import BaseModel


class TrainModelDTO(BaseModel):
    dataset_id: int
    target_column: str
    model_name: str


class PredictionDTO(BaseModel):
    model_id: int
    data: dict