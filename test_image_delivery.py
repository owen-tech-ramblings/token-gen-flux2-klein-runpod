import base64
import hashlib
import io
import json
import time
import unittest
from unittest.mock import Mock, patch
from PIL import Image
import image_delivery as delivery


class ImageDeliveryTests(unittest.TestCase):
    def capability(self):
        value='a'*32
        return {'id':value,'url':delivery.ORIGIN+'/api/image/worker-transfers/'+value,
                'token':str(int(time.time())+3600)+'.'+'b'*64,'chunk_bytes':delivery.CHUNK_BYTES,
                'max_bytes':delivery.MAX_BYTES,'max_images':4}

    def test_wrapper_returns_compact_original_png_descriptors(self):
        cap=self.capability();buf=io.BytesIO();Image.new('RGBA',(32,32),(1,2,3,123)).save(buf,format='PNG')
        raw=buf.getvalue(); uploaded={}
        def request(transfer,method,suffix,body):
            self.assertEqual(transfer,cap)
            if method=='PUT': uploaded[suffix]=body;return {'ok':True}
            index=int(suffix.split('/')[1]);manifest=json.loads(body)
            self.assertEqual(manifest['provider_id'],'provider-job')
            self.assertEqual(manifest['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(b''.join(v for k,v in uploaded.items() if k.startswith(f'/{index}/')),raw)
            return {'ok':True,'image':{'type':'api_transfer','id':cap['id'],'index':index,'filename':f'image_{index}.png','bytes':len(raw),'sha256':manifest['sha256']}}
        original=Mock(return_value={'images':[{'type':'base64','data':base64.b64encode(raw).decode(),'filename':'native.png'}]*4})
        with patch.object(delivery,'request',side_effect=request): result=delivery.wrap_handler(original)({'id':'provider-job','input':{'image_transfer':cap}})
        self.assertEqual(len(result['images']),4);self.assertLess(len(json.dumps(result)),2000)
        self.assertNotIn('data',result['images'][0]);self.assertNotIn('token',json.dumps(result))
        with patch.object(delivery,'request',side_effect=delivery.DeliveryError('upload failed')):
            self.assertEqual(delivery.wrap_handler(original)({'id':'provider-job','input':{'image_transfer':cap}}),{'error':'upload failed'})

    def test_delivery_rejects_untrusted_destination_and_false_ack(self):
        cap=self.capability(); original=Mock()
        for change in [{'url':'https://other.test/upload'},{'url':cap['url']+'?token=x'},{'id':'../escape'},{'token':'1000000000.'+'b'*64},{'chunk_bytes':1},{'max_images':5}]:
            self.assertIn('error',delivery.wrap_handler(original)({'id':'provider-job','input':{'image_transfer':{**cap,**change}}}))
        original.assert_not_called()
        buf=io.BytesIO();Image.new('RGB',(8,8),'red').save(buf,format='PNG')
        result={'images':[{'type':'base64','data':base64.b64encode(buf.getvalue()).decode()}]}
        with patch.object(delivery,'request',return_value={'ok':True,'image':{'type':'api_transfer'}}):
            with self.assertRaises(delivery.DeliveryError):delivery.deliver(result,cap,"provider-job")
        with patch.object(delivery,'build_opener') as opener:
            response=opener.return_value.open.return_value.__enter__.return_value
            response.status=200;response.read.return_value=b'{"ok": true}'
            self.assertTrue(delivery.request(cap,'PUT','/0/chunks/0',b'png')['ok'])
            sent=opener.return_value.open.call_args.args[0]
            self.assertEqual(sent.full_url,cap['url']+'/0/chunks/0')
            self.assertEqual(sent.get_header('Authorization'),'Bearer '+cap['token'])
        self.assertIsNone(delivery.NoRedirect().redirect_request(None,None,302,'',{},'https://other.test'))

if __name__=='__main__':unittest.main()
