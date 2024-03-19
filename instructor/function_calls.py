from typing import Any, Dict, Optional, Type, TypeVar
from docstring_parser import parse
from functools import wraps
from pydantic import BaseModel, create_model, BaseConfig
from instructor.exceptions import IncompleteOutputException
from openai.types.chat import ChatCompletion
from instructor.mode import Mode
from instructor.utils import extract_json_from_codeblock
import logging

T = TypeVar("T")

logger = logging.getLogger("instructor")
from loguru import logger as alogger
# From json Schema handle validation downstream and also handle retries downstream
class OpenAISchemaFromJson(BaseModel):


    schema: Optional[Dict[str, Any]]
    name: str

    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        
        schema = cls.schema
        
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.name}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }
    @classmethod
    def from_response(
        cls,
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> str:
        message = completion.choices[0].message

        tool_call = message.tool_calls[0]  # type: ignore
        # assert (
        #         tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
        #     ), "Tool name does not match"
        function_call = tool_call.function.arguments
        alogger.info(function_call)
        return function_call



        
   

class OpenAISchema(BaseModel):  # type: ignore[misc]
    @classmethod  # type: ignore[misc]
    @property
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        if not isinstance(cls, dict[str,any]):
            schema = cls.model_json_schema()
        else:
            schema = cls.schema
        
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }
    @classmethod
    def from_response(
        cls,
        completion: ChatCompletion,
        validation_context: Optional[Dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> BaseModel:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected
            validation_context (dict): The validation context to use for validating the response
            strict (bool): Whether to use strict json parsing
            mode (Mode): The openai completion mode

        Returns:
            cls (OpenAISchema): An instance of the class
        """
        assert hasattr(completion, "choices")

        if completion.choices[0].finish_reason == "length":
            logger.error("Incomplete output detected, should increase max_tokens")
            raise IncompleteOutputException()

        # If Anthropic, this should be different
        message = completion.choices[0].message

        if mode == Mode.FUNCTIONS:
            assert (
                message.function_call.name == cls.openai_schema["name"]  # type: ignore[index]
            ), "Function name does not match"
            model_response = cls.model_validate_json(
                message.function_call.arguments,  # type: ignore[attr-defined]
                context=validation_context,
                strict=strict,
            )
        elif mode in {Mode.TOOLS, Mode.MISTRAL_TOOLS}:
            assert (
                len(message.tool_calls or []) == 1
            ), "Instructor does not support multiple tool calls, use List[Model] instead."
            tool_call = message.tool_calls[0]  # type: ignore
            assert (
                tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
            ), "Tool name does not match"
            model_response = cls.model_validate_json(
                tool_call.function.arguments,
                context=validation_context,
                strict=strict,
            )
        elif mode in {Mode.JSON, Mode.JSON_SCHEMA, Mode.MD_JSON}:
            if mode == Mode.MD_JSON:
                message.content = extract_json_from_codeblock(message.content or "")

            model_response = cls.model_validate_json(
                message.content,  # type: ignore
                context=validation_context,
                strict=strict,
            )
        else:
            raise ValueError(f"Invalid patch mode: {mode}")

        # TODO: add logging or response handler
        return model_response


def openai_schema(cls: Type[BaseModel]) -> OpenAISchema:
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    return wraps(cls, updated=())(
        create_model(
            cls.__name__,
            __base__=(cls, OpenAISchema),
        )
    )  # type: ignore[all]

def openai_schema_from_json(json_schema: Dict[str, Any]) -> OpenAISchemaFromJson:
    schema = json_schema
    # Create openaiSchema with json_schame
    class_name = json_schema['title']
    # alogger.info(class_name)
    # # Define a custom configuration class that allows arbitrary types
    # class AllowArbitraryTypesConfig(BaseConfig):
    #     arbitrary_types_allowed = True

    # return OpenAISchemaFromJson(schema=schema, name=class_name)
     # Define a custom configuration class that allows arbitrary types
    return OpenAISchemaFromJson(schema=schema, name=class_name)
    


