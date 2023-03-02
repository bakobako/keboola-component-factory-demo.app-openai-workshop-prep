import logging
import json
import openai
from typing import Iterator, List
from csv import DictReader, DictWriter

from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException

# configuration variables
KEY_API_TOKEN = '#api_token'
KEY_PROMPT = 'prompt'
KEY_TEXT_COLUMN = "text_column"

REQUIRED_PARAMETERS = [KEY_API_TOKEN, KEY_PROMPT]
REQUIRED_IMAGE_PARS = []

MODEL_NAME = "text-davinci-003"
MODEL_BASE_TEMPERATURE = 0.7
MODEL_BASE_MAX_TOKENS = 512
MODEL_BASE_TOP_P = 1
MODEL_BASE_FREQUENCY_PENALTY = 0
MODEL_BASE_PRESENCE_PENALTY = 0


def read_messages_from_file(file_name: str, file_columns: List[str], text_column: str) -> Iterator[str]:
    with open(file_name) as in_file:
        reader = DictReader(in_file, file_columns)
        for line in reader:
            yield line.get(text_column)


def process_message(openai_key: str, prompt: str) -> str:
    openai.api_key = openai_key
    response = openai.Completion.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=MODEL_BASE_TEMPERATURE,
        max_tokens=MODEL_BASE_MAX_TOKENS,
        top_p=MODEL_BASE_TOP_P,
        frequency_penalty=MODEL_BASE_FREQUENCY_PENALTY,
        presence_penalty=MODEL_BASE_PRESENCE_PENALTY
    )
    return response.choices[0].text


def analyze_messages_in_file(in_file_name: str, text_column: str, file_columns: List[str], out_file_name: str,
                             base_prompt: str, openai_key: str) -> None:
    with open(out_file_name, 'w') as out_file:
        writer = DictWriter(out_file, ["message", "output"])
        for message in read_messages_from_file(in_file_name, file_columns, text_column):
            prompt = f"{base_prompt}\n\"\"\"{message}\"\"\""
            data = json.loads(process_message(openai_key, prompt))
            writer.writerow({"message": message, "output": data})


class Component(ComponentBase):
    def __init__(self):
        super().__init__()

    def run(self):
        self.validate_configuration_parameters(REQUIRED_PARAMETERS)
        self.validate_image_parameters(REQUIRED_IMAGE_PARS)
        params = self.configuration.parameters

        base_prompt = params.get(KEY_PROMPT)
        api_token = params.get(KEY_API_TOKEN)
        text_column = params.get(KEY_TEXT_COLUMN)

        input_table = self.get_input_tables_definitions()[0]

        output_table = self.create_out_table_definition("analyzed_output.csv")

        analyze_messages_in_file(input_table.full_path, text_column, input_table.columns, output_table.full_path,
                                 base_prompt, api_token)

        self.write_manifest(output_table)


"""
        Main entrypoint
"""
if __name__ == "__main__":
    try:
        comp = Component()
        # this triggers the run method by default and is controlled by the configuration.action parameter
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
