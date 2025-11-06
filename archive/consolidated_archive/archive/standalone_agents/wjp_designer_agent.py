#!/usr/bin/env python3
"""
WJP Designer Agent - Prompt to Image Generation
==============================================

This agent generates design images from prompts and creates metadata
for the next stage in the pipeline.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import random

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from wjp_file_manager import WJPFileManager, JobMetadata, ProcessStage, MaterialCode

class DesignerAgent:
    """Designer Agent for generating images from prompts."""
    
    def __init__(self):
        self.file_manager = WJPFileManager()
        self.output_dir = Path("output/designer")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Design templates for different categories
        self.design_templates = {
            "Inlay Tile": self._create_inlay_tile_design,
            "Medallion": self._create_medallion_design,
            "Border": self._create_border_design,
            "Jali Panel": self._create_jali_design,
            "Drainage Cover": self._create_drainage_design,
            "Nameplate": self._create_nameplate_design
        }
        
        # Material color schemes
        self.material_colors = {
            "Tan Brown Granite": (139, 69, 19),    # Brown
            "Marble": (255, 255, 255),            # White
            "Stainless Steel": (192, 192, 192),   # Silver
            "Aluminum": (169, 169, 169),          # Gray
            "Brass": (205, 133, 63),              # Gold
            "Generic": (128, 128, 128)            # Gray
        }
    
    def run(self, job_id: str, prompt: str, material: str, thickness_mm: int, 
            category: str = "Inlay Tile", dimensions_inch: list = [24, 24],
            cut_spacing_mm: float = 3.0, min_radius_mm: float = 2.0) -> Tuple[str, str]:
        """
        Generate design image from prompt and create metadata.
        
        Args:
            job_id: Unique job identifier
            prompt: Design prompt
            material: Material type
            thickness_mm: Material thickness
            category: Design category
            dimensions_inch: Design dimensions [width, height]
            cut_spacing_mm: Minimum cut spacing
            min_radius_mm: Minimum radius
            
        Returns:
            Tuple of (image_path, metadata_path)
        """
        print(f"🎨 **Designer Agent - Processing Job: {job_id}**")
        print(f"   Prompt: {prompt}")
        print(f"   Material: {material}")
        print(f"   Category: {category}")
        
        # Extract design code from job_id
        design_code = job_id
        
        # Generate design image
        image_path = self._generate_design_image(
            design_code=design_code,
            material=material,
            category=category,
            dimensions_inch=dimensions_inch,
            prompt=prompt
        )
        
        # Create metadata
        metadata = JobMetadata(
            design_code=design_code,
            material=material,
            thickness_mm=thickness_mm,
            category=category,
            dimensions_inch=dimensions_inch,
            cut_spacing_mm=cut_spacing_mm,
            min_radius_mm=min_radius_mm,
            prompt_used=prompt,
            next_stage="image_to_dxf",
            timestamp=datetime.now().isoformat()
        )
        
        # Save metadata
        metadata_path = self.file_manager.save_metadata(design_code, metadata, "designer")
        
        print(f"✅ **Designer Agent Complete**")
        print(f"   Image: {os.path.basename(image_path)}")
        print(f"   Metadata: {os.path.basename(metadata_path)}")
        
        return image_path, metadata_path
    
    def _generate_design_image(self, design_code: str, material: str, 
                              category: str, dimensions_inch: list, prompt: str) -> str:
        """Generate design image based on category and prompt."""
        
        # Get template function
        template_func = self.design_templates.get(category, self._create_inlay_tile_design)
        
        # Generate image
        image = template_func(material, dimensions_inch, prompt)
        
        # Generate filename
        material_code = self._get_material_code(material)
        filename = self.file_manager.generate_filename(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=25,  # Default thickness for design stage
            process_stage=ProcessStage.DESIGN,
            version="V1",
            extension="png"
        )
        
        # Get file path
        file_path = self.file_manager.get_file_path(
            design_code=design_code,
            material_code=material_code,
            thickness_mm=25,
            process_stage=ProcessStage.DESIGN,
            stage_folder="designer",
            version="V1",
            extension="png"
        )
        
        # Save image
        image.save(file_path)
        
        return file_path
    
    def _get_material_code(self, material: str) -> str:
        """Get material code from material name."""
        material_mapping = {
            "Tan Brown Granite": "TANB",
            "Marble": "MARB",
            "Stainless Steel": "STST",
            "Aluminum": "ALUM",
            "Brass": "BRAS",
            "Generic": "GENE"
        }
        return material_mapping.get(material, "GENE")
    
    def _create_inlay_tile_design(self, material: str, dimensions_inch: list, prompt: str) -> Image.Image:
        """Create inlay tile design."""
        width, height = dimensions_inch
        # Convert inches to pixels (100 DPI)
        img_width, img_height = width * 100, height * 100
        
        # Create base image
        base_color = self.material_colors.get(material, (128, 128, 128))
        image = Image.new('RGB', (img_width, img_height), base_color)
        draw = ImageDraw.Draw(image)
        
        # Draw inlay pattern
        center_x, center_y = img_width // 2, img_height // 2
        
        # Outer border
        border_width = 20
        draw.rectangle([border_width, border_width, img_width - border_width, img_height - border_width], 
                      outline=(255, 255, 255), width=3)
        
        # Central medallion
        medallion_radius = min(img_width, img_height) // 4
        draw.ellipse([center_x - medallion_radius, center_y - medallion_radius,
                     center_x + medallion_radius, center_y + medallion_radius],
                    outline=(255, 255, 255), width=2)
        
        # Corner decorations
        corner_size = 30
        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                corner_x = center_x + x_offset * (img_width // 3)
                corner_y = center_y + y_offset * (img_height // 3)
                draw.ellipse([corner_x - corner_size, corner_y - corner_size,
                            corner_x + corner_size, corner_y + corner_size],
                           outline=(255, 255, 255), width=2)
        
        return image
    
    def _create_medallion_design(self, material: str, dimensions_inch: list, prompt: str) -> Image.Image:
        """Create medallion design."""
        width, height = dimensions_inch
        img_width, img_height = width * 100, height * 100
        
        base_color = self.material_colors.get(material, (128, 128, 128))
        image = Image.new('RGB', (img_width, img_height), base_color)
        draw = ImageDraw.Draw(image)
        
        center_x, center_y = img_width // 2, img_height // 2
        
        # Concentric circles
        for radius in range(50, min(img_width, img_height) // 2, 30):
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        outline=(255, 255, 255), width=2)
        
        # Central pattern
        draw.ellipse([center_x - 20, center_y - 20,
                     center_x + 20, center_y + 20],
                    fill=(255, 255, 255))
        
        return image
    
    def _create_border_design(self, material: str, dimensions_inch: list, prompt: str) -> Image.Image:
        """Create border design."""
        width, height = dimensions_inch
        img_width, img_height = width * 100, height * 100
        
        base_color = self.material_colors.get(material, (128, 128, 128))
        image = Image.new('RGB', (img_width, img_height), base_color)
        draw = ImageDraw.Draw(image)
        
        # Border pattern
        border_width = 40
        draw.rectangle([border_width, border_width, img_width - border_width, img_height - border_width], 
                      outline=(255, 255, 255), width=3)
        
        # Corner decorations
        corner_size = 20
        corners = [
            (border_width + corner_size, border_width + corner_size),
            (img_width - border_width - corner_size, border_width + corner_size),
            (border_width + corner_size, img_height - border_width - corner_size),
            (img_width - border_width - corner_size, img_height - border_width - corner_size)
        ]
        
        for corner_x, corner_y in corners:
            draw.ellipse([corner_x - corner_size, corner_y - corner_size,
                         corner_x + corner_size, corner_y + corner_size],
                        outline=(255, 255, 255), width=2)
        
        return image
    
    def _create_jali_design(self, material: str, dimensions_inch: list, prompt: str) -> Image.Image:
        """Create jali panel design."""
        width, height = dimensions_inch
        img_width, img_height = width * 100, height * 100
        
        base_color = self.material_colors.get(material, (128, 128, 128))
        image = Image.new('RGB', (img_width, img_height), base_color)
        draw = ImageDraw.Draw(image)
        
        # Grid pattern for jali
        grid_size = 30
        for x in range(grid_size, img_width, grid_size):
            draw.line([(x, 0), (x, img_height)], fill=(255, 255, 255), width=1)
        
        for y in range(grid_size, img_height, grid_size):
            draw.line([(0, y), (img_width, y)], fill=(255, 255, 255), width=1)
        
        # Add some circular cutouts
        for _ in range(20):
            x = random.randint(50, img_width - 50)
            y = random.randint(50, img_height - 50)
            radius = random.randint(10, 20)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        fill=(255, 255, 255))
        
        return image
    
    def _create_drainage_design(self, material: str, dimensions_inch: list, prompt: str) -> Image.Image:
        """Create drainage cover design."""
        width, height = dimensions_inch
        img_width, img_height = width * 100, height * 100
        
        base_color = self.material_colors.get(material, (128, 128, 128))
        image = Image.new('RGB', (img_width, img_height), base_color)
        draw = ImageDraw.Draw(image)
        
        # Border
        border_width = 20
        draw.rectangle([border_width, border_width, img_width - border_width, img_height - border_width], 
                      outline=(255, 255, 255), width=3)
        
        # Drainage slots
        slot_width = 10
        slot_length = 80
        slot_spacing = 30
        
        for y in range(border_width + 30, img_height - border_width - 30, slot_spacing):
            draw.rectangle([img_width // 2 - slot_length // 2, y,
                           img_width // 2 + slot_length // 2, y + slot_width],
                          fill=(255, 255, 255))
        
        return image
    
    def _create_nameplate_design(self, material: str, dimensions_inch: list, prompt: str) -> Image.Image:
        """Create nameplate design."""
        width, height = dimensions_inch
        img_width, img_height = width * 100, height * 100
        
        base_color = self.material_colors.get(material, (128, 128, 128))
        image = Image.new('RGB', (img_width, img_height), base_color)
        draw = ImageDraw.Draw(image)
        
        # Border
        border_width = 15
        draw.rectangle([border_width, border_width, img_width - border_width, img_height - border_width], 
                      outline=(255, 255, 255), width=2)
        
        # Text area (simplified as rectangle for now)
        text_area_width = img_width - 2 * border_width - 20
        text_area_height = img_height - 2 * border_width - 20
        text_x = border_width + 10
        text_y = border_width + 10
        
        draw.rectangle([text_x, text_y, text_x + text_area_width, text_y + text_area_height],
                      outline=(255, 255, 255), width=1)
        
        return image

def test_designer_agent():
    """Test the Designer Agent."""
    print("🎨 **Testing Designer Agent**")
    print("=" * 50)
    
    # Create designer agent
    designer = DesignerAgent()
    
    # Test cases
    test_cases = [
        {
            "job_id": "SR06",
            "prompt": "Waterjet-safe Tan Brown granite tile with white marble inlay, 24x24 inch",
            "material": "Tan Brown Granite",
            "thickness_mm": 25,
            "category": "Inlay Tile",
            "dimensions_inch": [24, 24]
        },
        {
            "job_id": "MD01",
            "prompt": "Circular medallion design for granite flooring",
            "material": "Marble",
            "thickness_mm": 20,
            "category": "Medallion",
            "dimensions_inch": [36, 36]
        },
        {
            "job_id": "JL02",
            "prompt": "Jali panel with geometric perforations",
            "material": "Stainless Steel",
            "thickness_mm": 3,
            "category": "Jali Panel",
            "dimensions_inch": [48, 36]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 **Test Case {i}: {test_case['job_id']}**")
        
        try:
            image_path, metadata_path = designer.run(**test_case)
            
            print(f"   ✅ Image generated: {os.path.basename(image_path)}")
            print(f"   ✅ Metadata created: {os.path.basename(metadata_path)}")
            
            # Verify files exist
            if os.path.exists(image_path) and os.path.exists(metadata_path):
                print(f"   ✅ Files verified successfully")
            else:
                print(f"   ❌ File verification failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 **Designer Agent Test Completed!**")
    
    return designer

if __name__ == "__main__":
    test_designer_agent()
