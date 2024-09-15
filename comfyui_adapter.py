# This is an adapter for ComfyUI with a custom UI client that enables controlling the UI from Python
# This is more convenient than using the API directly since you don't have to export your workflow to JSON
# It's great for iterative testing and debugging!
# --------------------------------------------------------------------------------

import asyncio
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Union, List
from urllib.error import HTTPError
import urllib.request
from PIL import Image
import numpy as np
from websocket import WebSocket, WebSocketTimeoutException


# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('comfyui')

class ComfyUIAdapter:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.freed = False

    def initialize_ws(self):
        self.client_id = str(uuid.uuid4())
        self.ws = WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        self.ws.timeout = 1
        self.freed = False

    def free(self):
        self.freed = True

    def json_post(self, url: str, data: Dict = None, verbose: bool = False) -> Dict:
        data = data or {}
        data['verbose'] = verbose
        data['client_id'] = self.client_id
        encoded_data = json.dumps(data).encode('utf-8')

        address = f'http://{self.server_address}{url}' if not url.startswith('http') else url

        req = urllib.request.Request(
            address,
            headers={'Content-Type': 'application/json'},
            data=encoded_data
        )

        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read())
        except HTTPError as e:
            log.error(f"HTTP Error: {str(e)}")
            log.error("- Is the server running?")
            log.error("- Is the uiapi plugin OK?")
            return None

    async def query_fields(self, verbose: bool = False) -> Dict:
        response = self.json_post('/uiapi/query_fields', verbose=verbose)
        if isinstance(response, dict):
            return response.get('result') or response.get('response')
        raise ValueError("Unexpected response format from server")

    def get_field(self, path_or_paths: Union[str, List[str]], verbose: bool = False) -> Union[Any, Dict[str, Any]]:
        is_single = isinstance(path_or_paths, str)
        paths = [path_or_paths] if is_single else path_or_paths

        response = self.json_post('/uiapi/get_fields', {"fields": paths}, verbose=verbose)

        if isinstance(response, dict):
            result = response.get('result') or response.get('response')
            return result[path_or_paths] if is_single else result
        raise ValueError("Unexpected response format from server")

    def set(self, path_or_fields: Union[str, List[tuple]], value: Any = None, verbose: bool = False):
        fields = [(path_or_fields, value)] if isinstance(path_or_fields, str) else path_or_fields
        processed_fields = []

        for path, val in fields:
            if isinstance(val, (Image.Image, np.ndarray)):
                self.set_img(path, val, verbose)
            else:
                processed_fields.append([path, val])

        if processed_fields:
            return self.json_post('/uiapi/set_fields', {"fields": processed_fields}, verbose=verbose)

    def set_img(self, path: str, value: Union[Image.Image, np.ndarray], verbose: bool = False):
        api = str(uuid.uuid4())
        input_name = f'uiapi_{api}.png'
        self.set(path, input_name, verbose)

        if isinstance(value, Image.Image):
            img_byte_arr = io.BytesIO()
            value.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
        elif isinstance(value, np.ndarray):
            is_success, img_byte_arr = cv2.imencode(".png", value)
            if not is_success:
                raise ValueError("Failed to encode image")
            img_byte_arr = img_byte_arr.tobytes()

        data = {'image': img_byte_arr, 'name': input_name}
        self.json_post('/uiapi/upload_image', data, verbose=verbose)

    def connect(self, path1: str, path2: str, verbose: bool = False):
        return self.json_post('/uiapi/set_connection', {"field": [path1, path2]}, verbose=verbose)

    def execute(self, delete_output: bool = False, wait: bool = True):
        ret = self.json_post('/uiapi/execute')
        if not wait:
            return ret

        exec_id = ret['response']['prompt_id']
        self.await_execution()

        workflow_json = self.json_post('/uiapi/get_workflow')
        address = self.find_output_node(workflow_json['response'])
        history = self.get_history(exec_id)[exec_id]

        filenames = eval(f"history['outputs']{address}")['images']
        images = []
        for img_info in filenames:
            image_data = self.get_image(img_info['filename'], img_info['subfolder'], img_info['type'])
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
            if delete_output:
                # Implement deletion if needed
                pass

        return images[0]

    def await_execution(self):
        self.initialize_ws()
        while True:
            try:
                out = self.ws.recv()
                if isinstance(out, str):
                    msg = json.loads(out)
                    if msg['type'] == 'status' and msg['data']['status']['exec_info']['queue_remaining'] == 0:
                        return
            except WebSocketTimeoutException:
                pass
            except Exception as e:
                log.error(f"Error in await_execution: {str(e)}")
                return

            if self.freed:
                self.freed = False
                return

    def get_history(self, prompt_id: str) -> Dict:
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    @staticmethod
    def find_output_node(json_object: Dict) -> str:
        for key, value in json_object.items():
            if isinstance(value, dict):
                if value.get("class_type") in ["SaveImage", "Image Save"]:
                    return f"['{key}']"
                result = ComfyUIAdapter.find_output_node(value)
                if result:
                    return result
        return None

    def txt2img(self, **args):
        self.send_values(args)
        self.connect("ccg3.CONDITIONING", "KSampler.positive")
        return self.execute()

    def img2img(self, **args):
        self.send_values(args)
        self.connect("ConditioningAverage.CONDITIONING", "KSampler.positive")
        return self.execute()

    def send_values(self, args: Dict[str, Any]):
        for key, value in args.items():
            self.set(key, value)

# Example usage
if __name__ == "__main__":
    comfy = ComfyUIAdapter()
    result = comfy.txt2img(prompt="A beautiful landscape", steps=20, cfg_scale=7)
    result.save("output.png")
