import unittest
import os
import sys
from pathlib import Path
from moviepy import (
    VideoFileClip,
)
# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.models.schema import MaterialInfo
from app.services import video as vd
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

class TestVideoService(unittest.TestCase):
    def setUp(self):
        self.test_img_path = os.path.join(resources_dir, "1.png")
    
    def tearDown(self):
        pass
    
    def test_preprocess_video(self):
        if not os.path.exists(self.test_img_path):
            self.fail(f"test image not found: {self.test_img_path}")
        
        # test preprocess_video function
        m = MaterialInfo()
        m.url = self.test_img_path
        m.provider = "local"
        print(m)
        
        materials = vd.preprocess_video([m], clip_duration=4)
        print(materials)
        
        # verify result
        self.assertIsNotNone(materials)
        self.assertEqual(len(materials), 1)
        self.assertTrue(materials[0].url.endswith(".mp4"))
        
        # moviepy get video info
        clip = VideoFileClip(materials[0].url)
        print(clip)
        
        # clean generated test video file
        if os.path.exists(materials[0].url):
            os.remove(materials[0].url)
    
    def test_wrap_text(self):
        """test text wrapping function"""
        try:
            font_path = os.path.join(utils.font_dir(), "STHeitiMedium.ttc")
            if not os.path.exists(font_path):
                self.fail(f"font file not found: {font_path}")
                
            # test english text wrapping
            test_text_en = "This is a sample text content for testing English long sentence line breaks, which should wrap based on width constraints"
            
            wrapped_text_en, text_height_en = vd.wrap_text(
                text=test_text_en,
                max_width=300,
                font=font_path,
                fontsize=30
            )
            print(wrapped_text_en, text_height_en)
            # verify text is wrapped
            self.assertIn("\n", wrapped_text_en)
            
            # test chinese text wrapping
            test_text_zh = "This is a sample text content for testing Chinese long sentence line breaks, which should wrap based on width constraints"
            wrapped_text_zh, text_height_zh = vd.wrap_text(
                text=test_text_zh,
                max_width=300,
                font=font_path,
                fontsize=30
            )   
            print(wrapped_text_zh, text_height_zh)
            # verify chinese text is wrapped
            self.assertIn("\n", wrapped_text_zh)
        except Exception as e:
            self.fail(f"test wrap_text failed: {str(e)}")

if __name__ == "__main__":
    unittest.main() 