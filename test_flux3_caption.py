import json
import unittest


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
        clip=Clip(); a,b=object(),object(); boxes=[{'id':'racer','bbox':[100,200,900,800],'desc':'racer'}]
        node=TokenGenFlux3Caption()
        result=node.caption('A reptilian girl racer',json.dumps(boxes),True,4096,2304,clip=clip,image0=a,image1=b)[0]
        self.assertEqual(clip.options['images'],[a,b])
        self.assertFalse(clip.options['thinking'])
        self.assertFalse(clip.generation['do_sample'])
        self.assertIn('"bbox": [100, 200, 900, 800]',result)
        self.assertIn('<racer>',result)
        self.assertEqual(node.caption('Already expanded caption','[]',False,1024,1024)[0],'Already expanded caption')
        clip.decode=lambda ids:''
        with self.assertRaisesRegex(RuntimeError,'empty'):
            node.caption('racer','[]',True,1024,1024,clip=clip)
