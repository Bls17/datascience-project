from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from src.datascience.pipeline.data_validation_pipeline import (
    DataValidationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.initiate_data_ingestion()
    logger.info(
        f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<\n\nx==============x"
    )
except Exception as e:
    logger.exception(e)
    raise (e)


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"Stage {STAGE_NAME} started")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e
