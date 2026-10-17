"""Upload original PNG bytes outside RunPod's bounded JSON result envelope."""
import base64
import hashlib
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

CHUNK_BYTES = 8 * 1024 * 1024
MAX_BYTES = 256 * 1024 * 1024
ORIGIN = 'https://token-gen-api.owenonthenet.com'


class DeliveryError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate(value):
    if not isinstance(value, dict):
        raise DeliveryError('Missing native image delivery capability.')
    identifier, url, token = value.get('id'), value.get('url'), value.get('token')
    if (not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier)
            or url != f'{ORIGIN}/api/image/worker-transfers/{identifier}'
            or not isinstance(token, str) or not re.fullmatch(r'[0-9]{10}\.[a-f0-9]{64}', token)
            or int(token.split('.')[0]) <= time.time()
            or type(value.get('chunk_bytes')) is not int or value['chunk_bytes'] != CHUNK_BYTES
            or type(value.get('max_bytes')) is not int or not 0 < value['max_bytes'] <= MAX_BYTES
            or type(value.get('max_images')) is not int or not 0 < value['max_images'] <= 4):
        raise DeliveryError('Invalid native image delivery capability.')
    return value


def request(transfer, method, suffix, body):
    opener = build_opener(NoRedirect())
    for attempt in range(3):
        validate(transfer)
        req = Request(transfer['url'] + suffix, method=method, data=body,
                      headers={'Authorization': 'Bearer ' + transfer['token'],
                               'Content-Type': 'application/octet-stream' if method == 'PUT' else 'application/json',
                               'User-Agent': 'Token-Gen-ImageTransfer/1.0'})
        try:
            with opener.open(req, timeout=90) as response:
                raw = response.read(4097)
                if response.status != 200 or len(raw) > 4096:
                    raise DeliveryError('Native image delivery returned an invalid acknowledgement.')
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get('ok') is not True:
                    raise DeliveryError('Native image delivery was not acknowledged.')
                return data
        except HTTPError as exc:
            retryable = exc.code in (408,429) or 500 <= exc.code <= 599
            exc.close()
            if not retryable or attempt == 2:
                raise DeliveryError('Native image delivery HTTP request failed.') from None
        except (URLError, OSError):
            if attempt == 2:
                raise DeliveryError('Native image delivery connection failed.') from None
        except (ValueError, UnicodeError):
            raise DeliveryError('Native image delivery acknowledgement was invalid.') from None
        time.sleep(attempt + 1)
    raise DeliveryError('Native image delivery failed.')


def deliver(result, transfer, provider_id):
    validate(transfer)
    if not isinstance(provider_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", provider_id):
        raise DeliveryError("Invalid provider job ID for native image delivery.")
    if not isinstance(result, dict) or result.get('error'):
        return result
    images = result.get('images')
    if not isinstance(images, list) or not 0 < len(images) <= transfer['max_images']:
        raise DeliveryError('Native image generation returned an invalid output count.')
    delivered = []
    for index, image in enumerate(images):
        if (not isinstance(image, dict) or image.get('type') != 'base64'
                or not isinstance(image.get('data'), str)
                or len(image['data']) > ((transfer['max_bytes'] + 2) // 3) * 4):
            raise DeliveryError('Native image output format is unsupported.')
        try:
            raw = base64.b64decode(image['data'], validate=True)
        except (ValueError, TypeError):
            raise DeliveryError('Native image output encoding is invalid.') from None
        if not 0 < len(raw) <= transfer['max_bytes'] or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise DeliveryError('Native image output must be a lossless PNG.')
        chunks = (len(raw) + CHUNK_BYTES - 1) // CHUNK_BYTES
        manifest = {'provider_id':provider_id, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'chunks': chunks}
        for chunk, start in enumerate(range(0, len(raw), CHUNK_BYTES)):
            request(transfer,'PUT',f'/{index}/chunks/{chunk}',raw[start:start+CHUNK_BYTES])
        ack = request(transfer,'POST',f'/{index}/complete',json.dumps(manifest).encode())
        expected = {'type':'api_transfer','id':transfer['id'],'index':index,'filename':f'image_{index}.png',
                    'bytes':manifest['bytes'],'sha256':manifest['sha256']}
        if ack.get('image') != expected:
            raise DeliveryError('Native image completion acknowledgement did not match the original.')
        delivered.append(expected)
    return {**result, 'images': delivered}


def wrap_handler(handler):
    def wrapped(job):
        try:
            transfer = validate(job.get('input', {}).get('image_transfer'))
            return deliver(handler(job), transfer, job.get("id"))
        except DeliveryError as exc:
            return {'error': str(exc)}
    return wrapped
