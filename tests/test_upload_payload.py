import base64
import json
import unittest
from http import HTTPStatus

from map_boundary_builder.upload_payload import UploadPayloadError, parse_json_upload_body


class JsonUploadPayloadTests(unittest.TestCase):
    def test_parse_json_upload_body_decodes_image_and_fields(self) -> None:
        image_bytes = b"II*\x00tiff-bytes"
        payload = {
            "image": {
                "filename": "../Tesla Dallas.tiff",
                "content_type": "image/tiff",
                "data_base64": base64.b64encode(image_bytes).decode("ascii"),
            },
            "fields": {
                "include_overlay": False,
                "city": "Dallas",
                "summary": {"ok": True},
            },
        }

        fields, files = parse_json_upload_body(
            json.dumps(payload).encode("utf-8"),
            max_upload_bytes=1024,
        )

        self.assertEqual(fields["include_overlay"], "0")
        self.assertEqual(fields["city"], "Dallas")
        self.assertEqual(fields["summary"], '{"ok":true}')
        self.assertEqual(files["image"], ("Tesla Dallas.tiff", image_bytes))

    def test_parse_json_upload_body_accepts_data_url(self) -> None:
        image_bytes = b"image-bytes"
        payload = {
            "image": {
                "filename": "map.tif",
                "data_base64": f"data:image/tiff;base64,{base64.b64encode(image_bytes).decode('ascii')}",
            }
        }

        _fields, files = parse_json_upload_body(
            json.dumps(payload).encode("utf-8"),
            max_upload_bytes=1024,
        )

        self.assertEqual(files["image"], ("map.tif", image_bytes))

    def test_parse_json_upload_body_rejects_invalid_base64(self) -> None:
        payload = {"image": {"filename": "map.tiff", "data_base64": "not-base64!!"}}

        with self.assertRaises(UploadPayloadError) as context:
            parse_json_upload_body(json.dumps(payload).encode("utf-8"), max_upload_bytes=1024)

        self.assertEqual(context.exception.status, HTTPStatus.BAD_REQUEST)

    def test_parse_json_upload_body_rejects_decoded_upload_over_limit(self) -> None:
        payload = {
            "image": {
                "filename": "map.tiff",
                "data_base64": base64.b64encode(b"12345").decode("ascii"),
            }
        }

        with self.assertRaises(UploadPayloadError) as context:
            parse_json_upload_body(json.dumps(payload).encode("utf-8"), max_upload_bytes=4)

        self.assertEqual(context.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


if __name__ == "__main__":
    unittest.main()
