import pytest
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

from instructor import patch
import instructor
from instructor.exceptions import IncompleteOutputException
from loguru import logger
@pytest.fixture
def instructor_client():
    return patch(OpenAI())

class StorageUnitInformation(BaseModel):
    price: int = Field(..., description="Price for the storage unit, in dollars.")
    square_feet: int = Field(..., description="Size of the storage unit, in square feet. ex) 25sqft")
    dimension: str = Field(..., description="Dimension ex) 5x5 of the storage unit.")
    metadata: str = Field(..., description="Additional metadata about the storage unit, such as location or features.")

class InformationList(BaseModel):
    storage_units: List[StorageUnitInformation] = Field(..., description="A list of all the different storage unit offerings on the given page including just the fields in the StorageUnitInformation model.")

# @pytest.mark.asyncio
# async def test_storage_unit_information_extraction(instructor_client):
#     system_prompt = "Please extract all the storage unit information from the following text."
#     user_prompt = "The first unit is a 5x5 unit priced at $25 per month. It's 25sqft and located in downtown. The second unit is a 10x10 unit priced at $50 per month. It's 100sqft and comes with climate control."
    
#     extracted_data = instructor_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         response_model=InformationList,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
#     )
#     logger.info(extracted_data)
#     assert len(extracted_data.storage_units) == 2
#     assert extracted_data.storage_units[0].price == 25
#     assert extracted_data.storage_units[0].square_feet == 25
#     assert extracted_data.storage_units[0].dimension == "5x5"
#     assert "downtown" in extracted_data.storage_units[0].metadata
#     assert extracted_data.storage_units[1].price == 50
#     assert extracted_data.storage_units[1].square_feet == 100
#     assert extracted_data.storage_units[1].dimension == "10x10"
#     assert "climate control" in extracted_data.storage_units[1].metadata

@pytest.mark.asyncio
async def test_storage_unit_information_extraction_from_json(instructor_client):
    # Json schema for the model
    system_prompt = "Please extract all the storage unit information from the following text."
    user_prompt = "The first unit is a 5x5 unit priced at $25 per month. It's 25sqft and located in downtown. The second unit is a 10x10 unit priced at $50 per month. It's 100sqft and comes with climate control."
    
    json_schema = InformationList.model_json_schema()
    logger.info(json_schema)
    extracted_data = instructor_client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=json_schema,
        messages=[
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}
         ]
    )
    logger.info(json_schema)
    logger.info(type(json_schema))
    