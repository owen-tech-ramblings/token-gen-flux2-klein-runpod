import json
import unittest
import torch


class CaptionTests(unittest.TestCase):
    def test_native_caption_preserves_references_layout_and_explicit_bypass(self):
        from token_gen_flux3 import TokenGenFlux3Caption
        class Clip:
            def tokenize(self,prompt,**kwargs):
                self.prompt=prompt;self.options=kwargs;return 'tokens'
            def generate(self,tokens,**kwargs):
                self.generation=kwargs;return 'generated'
            def decode(self,ids):
                return 'A green reptilian hoverboard rider <racer>. [{"id":"racer","bbox":[1,1,100,100],"desc":"green rider"}]'
        clip=Clip(); a=torch.zeros((1,1200,1600,3)); b=torch.ones((1,1600,1200,3)); boxes=[{'id':'racer','bbox':[100,200,900,800],'desc':'racer'}]
        node=TokenGenFlux3Caption()
        result=node.caption('A reptilian girl racer',json.dumps(boxes),True,4096,2304,clip=clip,image0=a,image1=b)[0]
        self.assertEqual(len(clip.options['images']),2)
        for resized in clip.options['images']:
            self.assertLessEqual(resized.shape[1]*resized.shape[2],724*724)
        self.assertGreater(clip.options['images'][0].shape[2],clip.options['images'][0].shape[1])
        self.assertLess(clip.options['images'][1].shape[2],clip.options['images'][1].shape[1])
        node.caption('racer','[]',True,4096,4096,clip=clip,**{f'image{i}':a for i in range(10)}, mask0=torch.ones((1,1200,1600)))
        self.assertEqual(len(clip.options['images']),10)
        for resized in clip.options['images']:
            self.assertLessEqual(resized.shape[1]*resized.shape[2],512*512)
        self.assertTrue(torch.all(clip.options['images'][0]==1))
        self.assertTrue(torch.all(clip.options['images'][1]==0))
        self.assertFalse(clip.options['thinking'])
        self.assertFalse(clip.generation['do_sample'])
        self.assertIn('"bbox": [100, 200, 900, 800]',result)
        self.assertIn('<racer>',result)
        self.assertEqual(node.caption('Already expanded caption','[]',False,1024,1024)[0],'Already expanded caption')
        clip.decode=lambda ids:''
        with self.assertRaisesRegex(RuntimeError,'empty'):
            node.caption('racer','[]',True,1024,1024,clip=clip)
