import json
from typing import Dict, List, Any


def load_from_file(file_path: str) -> Dict:
    """
    Reads a JSON file and returns parsed content.

    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict: Parsed JSON content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def parse_swagger_spec(swagger_spec: Dict) -> Dict[str, Any]:
    """
    Parses Swagger/OpenAPI spec and extracts endpoint information.

    Args:
        swagger_spec (dict): The Swagger specification.
    Returns:
        dict: Structured endpoint data for tool generation.
    """
    endpoints = {}
    paths = swagger_spec.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                operation_id = details.get("operationId", f"{method}_{path}").replace(" ", "_")

                endpoints[operation_id] = {
                    "path": path,
                    "method": method.upper(),
                    "description": details.get("description", "No description"),
                    "parameters": details.get("parameters", []),
                    "requestBody": details.get("requestBody", {}),
                    "responses": details.get("responses", {}),
                    "tags": details.get("tags", []),
                }

    return endpoints

