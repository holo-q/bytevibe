from typing import Tuple

import cv2
import numpy as np

def try_compression(img: np.ndarray, fmt: str, quality: int, width: int) -> \
	Tuple[bytes, int, int]:
	quality = int(round(quality))
	width = int(round(width))

	if width != img.shape[1]:
		resized_img = cv2.resize(img, (width, int(
			img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
	else:
		resized_img = img

	if fmt == "jpeg":
		_, buffer = cv2.imencode(".jpg", resized_img, [
			int(cv2.IMWRITE_JPEG_QUALITY), quality])
	elif fmt == "png":
		_, buffer = cv2.imencode(".png", resized_img, [
			int(cv2.IMWRITE_PNG_COMPRESSION), quality])
	elif fmt == "webp":
		_, buffer = cv2.imencode(".webp", resized_img, [
			int(cv2.IMWRITE_WEBP_QUALITY), quality])
	else:
		raise ValueError(f"Unsupported format: {fmt}")

	return buffer.tobytes(), quality, width
