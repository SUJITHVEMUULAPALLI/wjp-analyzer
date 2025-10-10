#!/usr/bin/env python3
"""
AI-Enhanced Image-to-DXF Converter
==================================

Combines local Ollama AI analysis with OpenAI API for prompt-to-image generation
with waterjet cutting considerations and live user interaction.
"""

import cv2
import numpy as np
import ezdxf
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import requests
import base64
from io import BytesIO
from PIL import Image

class AIEnhancedImageConverter:
    """AI-enhanced converter with Ollama local analysis and OpenAI integration."""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 openai_api_key: Optional[str] = None,
                 binary_threshold: int = 150,
                 min_area: int = 500,
                 dxf_size: int = 1000):
        """
        Initialize AI-enhanced converter.
        
        Args:
            ollama_url: Local Ollama server URL
            openai_api_key: OpenAI API key for image generation
            binary_threshold: Image processing threshold
            min_area: Minimum contour area
            dxf_size: Output DXF size
        """
        self.ollama_url = ollama_url
        self.openai_api_key = openai_api_key
        self.binary_threshold = binary_threshold
        self.min_area = min_area
        self.dxf_size = dxf_size
        
        # Waterjet cutting considerations
        self.min_kerf_width = 0.5  # mm
        self.min_feature_size = 2.0  # mm
        self.max_complexity = 1000  # vertices per shape
        
    def generate_image_from_prompt(self, prompt: str, style: str = "waterjet") -> Optional[str]:
        """
        Generate image from text prompt using OpenAI DALL-E with waterjet considerations.
        
        Args:
            prompt: Text description of desired design
            style: Style modifier (waterjet, geometric, organic, etc.)
            
        Returns:
            Path to generated image or None if failed
        """
        if not self.openai_api_key:
            return None
            
        # Enhance prompt with waterjet considerations
        enhanced_prompt = self._enhance_prompt_for_waterjet(prompt, style)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "dall-e-3",
                "prompt": enhanced_prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "standard",
                "style": "natural"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                image_url = result["data"][0]["url"]
                
                # Download and save image
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    timestamp = int(time.time())
                    filename = f"generated_design_{timestamp}.png"
                    filepath = os.path.join("output", "temp", filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    with open(filepath, "wb") as f:
                        f.write(img_response.content)
                    
                    return filepath
                    
        except Exception as e:
            print(f"Error generating image: {e}")
            
        return None
    
    def _enhance_prompt_for_waterjet(self, prompt: str, style: str) -> str:
        """Enhance prompt with waterjet cutting considerations."""
        waterjet_modifiers = {
            "waterjet": "clean geometric lines, suitable for waterjet cutting, minimal fine details",
            "geometric": "geometric patterns, straight lines, angular shapes, waterjet friendly",
            "organic": "flowing curves, natural shapes, simplified for cutting",
            "architectural": "architectural elements, clean lines, structural patterns"
        }
        
        style_modifier = waterjet_modifiers.get(style, waterjet_modifiers["waterjet"])
        
        enhanced = f"{prompt}, {style_modifier}, high contrast black and white, vector-style, "
        enhanced += "suitable for metal cutting, clear boundaries, no gradients"
        
        return enhanced
    
    def analyze_with_ollama(self, image_path: str, analysis_type: str = "waterjet") -> Dict[str, Any]:
        """
        Analyze image using local Ollama AI for waterjet considerations.
        
        Args:
            image_path: Path to image file
            analysis_type: Type of analysis (waterjet, complexity, optimization)
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Prepare analysis prompt
            prompt = self._get_analysis_prompt(analysis_type)
            
            payload = {
                "model": "llava",  # Vision-capable model
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get("response", "")
                
                # Parse analysis into structured data
                return self._parse_ollama_analysis(analysis_text, analysis_type)
                
        except Exception as e:
            print(f"Error in Ollama analysis: {e}")
            
        return {"error": "Analysis failed", "recommendations": []}
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Get appropriate analysis prompt for Ollama."""
        prompts = {
            "waterjet": """
            Analyze this image for waterjet cutting suitability. Consider:
            1. Line thickness and spacing (minimum 0.5mm kerf width)
            2. Feature complexity and detail level
            3. Sharp corners and acute angles
            4. Overall cutting feasibility
            5. Suggested simplifications
            
            Provide specific recommendations for optimization.
            """,
            "complexity": """
            Analyze the geometric complexity of this design:
            1. Number of distinct shapes
            2. Curve complexity and vertex count
            3. Nested or overlapping elements
            4. Overall processing difficulty
            
            Rate complexity from 1-10 and suggest simplifications.
            """,
            "optimization": """
            Suggest optimizations for this design:
            1. Remove unnecessary details
            2. Simplify complex curves
            3. Combine similar elements
            4. Improve cutting efficiency
            
            Provide specific actionable recommendations.
            """
        }
        
        return prompts.get(analysis_type, prompts["waterjet"])
    
    def _parse_ollama_analysis(self, analysis_text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse Ollama analysis text into structured data."""
        # Simple parsing - in production, use more sophisticated NLP
        recommendations = []
        complexity_score = 5  # Default
        
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('recommend' in line.lower() or 'suggest' in line.lower()):
                recommendations.append(line)
            elif 'complexity' in line.lower() and any(char.isdigit() for char in line):
                # Extract complexity score
                for char in line:
                    if char.isdigit():
                        complexity_score = int(char)
                        break
        
        return {
            "analysis_type": analysis_type,
            "complexity_score": complexity_score,
            "recommendations": recommendations,
            "raw_analysis": analysis_text,
            "timestamp": time.time()
        }
    
    def convert_with_ai_guidance(self, 
                                input_image: str, 
                                output_dxf: str,
                                ai_analysis: Optional[Dict] = None,
                                user_preferences: Optional[Dict] = None) -> Dict:
        """
        Convert image to DXF with AI-guided optimization.
        
        Args:
            input_image: Path to input image
            output_dxf: Path to output DXF
            ai_analysis: AI analysis results
            user_preferences: User customization preferences
            
        Returns:
            Conversion results with AI insights
        """
        try:
            # Load and preprocess image
            image = cv2.imread(input_image)
            if image is None:
                return {"error": "Could not load image", "polygons": 0}
            
            # Apply AI-guided preprocessing
            processed_image = self._ai_guided_preprocessing(image, ai_analysis, user_preferences)
            
            # Extract contours with AI-optimized parameters
            contours = self._extract_contours_ai_optimized(processed_image, ai_analysis)
            
            # Apply AI-recommended simplifications
            simplified_contours = self._apply_ai_simplifications(contours, ai_analysis)
            
            # Create DXF with waterjet considerations
            polygon_count = self._create_optimized_dxf(simplified_contours, output_dxf, ai_analysis)
            
            # Generate conversion report
            report = self._generate_conversion_report(
                input_image, output_dxf, polygon_count, ai_analysis, user_preferences
            )
            
            return report
            
        except Exception as e:
            return {"error": f"Conversion failed: {e}", "polygons": 0}
    
    def _ai_guided_preprocessing(self, image: np.ndarray, ai_analysis: Optional[Dict], user_preferences: Optional[Dict]) -> np.ndarray:
        """Apply AI-guided image preprocessing."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adjust parameters based on AI analysis
        if ai_analysis:
            complexity = ai_analysis.get("complexity_score", 5)
            # Higher complexity = more aggressive preprocessing
            if complexity > 7:
                # Apply stronger noise reduction
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                # Increase threshold for cleaner separation
                self.binary_threshold = min(200, self.binary_threshold + 20)
            elif complexity < 4:
                # Preserve more details for simple designs
                self.binary_threshold = max(100, self.binary_threshold - 20)
        
        # Apply user preferences
        if user_preferences:
            if user_preferences.get("preserve_details", False):
                self.binary_threshold = max(100, self.binary_threshold - 30)
            if user_preferences.get("aggressive_simplification", False):
                self.binary_threshold = min(200, self.binary_threshold + 30)
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)
        
        return binary
    
    def _extract_contours_ai_optimized(self, binary_image: np.ndarray, ai_analysis: Optional[Dict]) -> List:
        """Extract contours with AI-optimized parameters."""
        # Adjust contour detection based on AI analysis
        if ai_analysis and ai_analysis.get("complexity_score", 5) > 7:
            # Use more aggressive contour approximation for complex designs
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Standard contour detection
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        # Filter contours by area
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        
        return filtered_contours
    
    def _apply_ai_simplifications(self, contours: List, ai_analysis: Optional[Dict]) -> List:
        """Apply AI-recommended contour simplifications."""
        simplified_contours = []
        
        for contour in contours:
            # Calculate simplification factor based on AI analysis
            if ai_analysis:
                complexity = ai_analysis.get("complexity_score", 5)
                # Higher complexity = more simplification
                epsilon_factor = 0.02 + (complexity - 5) * 0.01
            else:
                epsilon_factor = 0.02
            
            # Apply Douglas-Peucker simplification
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Additional simplification for waterjet cutting
            if len(simplified) > self.max_complexity:
                # Further simplify if too complex
                epsilon = 0.05 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            simplified_contours.append(simplified)
        
        return simplified_contours
    
    def _create_optimized_dxf(self, contours: List, output_dxf: str, ai_analysis: Optional[Dict]) -> int:
        """Create DXF with waterjet cutting optimizations."""
        # Create DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        polygon_count = 0
        
        for contour in contours:
            if len(contour) < 3:
                continue
                
            # Convert contour to DXF polyline
            points = []
            for point in contour:
                x, y = point[0]
                # Scale to DXF coordinates
                x_scaled = (x / 1000.0) * self.dxf_size
                y_scaled = (y / 1000.0) * self.dxf_size
                points.append((x_scaled, y_scaled))
            
            # Close the polyline
            if len(points) > 2:
                points.append(points[0])
                
                # Create polyline entity
                polyline = msp.add_lwpolyline(points)
                polyline.closed = True
                polygon_count += 1
        
        # Save DXF
        doc.saveas(output_dxf)
        
        return polygon_count
    
    def _generate_conversion_report(self, 
                                  input_image: str, 
                                  output_dxf: str, 
                                  polygon_count: int,
                                  ai_analysis: Optional[Dict],
                                  user_preferences: Optional[Dict]) -> Dict:
        """Generate comprehensive conversion report."""
        report = {
            "input_image": input_image,
            "output_dxf": output_dxf,
            "polygon_count": polygon_count,
            "conversion_timestamp": time.time(),
            "ai_analysis": ai_analysis,
            "user_preferences": user_preferences,
            "waterjet_considerations": {
                "min_kerf_width": self.min_kerf_width,
                "min_feature_size": self.min_feature_size,
                "max_complexity": self.max_complexity
            },
            "processing_parameters": {
                "binary_threshold": self.binary_threshold,
                "min_area": self.min_area,
                "dxf_size": self.dxf_size
            }
        }
        
        return report
    
    def live_interactive_conversion(self, 
                                  input_image: str, 
                                  output_dxf: str,
                                  callback_func: Optional[callable] = None) -> Dict:
        """
        Perform live interactive conversion with user feedback.
        
        Args:
            input_image: Path to input image
            output_dxf: Path to output DXF
            callback_func: Function to call with progress updates
            
        Returns:
            Final conversion results
        """
        if callback_func:
            callback_func("Starting AI analysis...", 10)
        
        # Step 1: AI Analysis
        ai_analysis = self.analyze_with_ollama(input_image, "waterjet")
        
        if callback_func:
            callback_func("AI analysis complete. Applying optimizations...", 30)
        
        # Step 2: Initial conversion
        initial_result = self.convert_with_ai_guidance(input_image, output_dxf, ai_analysis)
        
        if callback_func:
            callback_func("Initial conversion complete. Generating preview...", 60)
        
        # Step 3: Generate preview and recommendations
        preview_path = output_dxf.replace('.dxf', '_preview.png')
        self._generate_preview(input_image, output_dxf, preview_path)
        
        if callback_func:
            callback_func("Conversion complete! Ready for review.", 100)
        
        # Add preview path to results
        initial_result["preview_path"] = preview_path
        initial_result["ai_recommendations"] = ai_analysis.get("recommendations", [])
        
        return initial_result
    
    def _generate_preview(self, input_image: str, output_dxf: str, preview_path: str):
        """Generate visual preview of conversion."""
        try:
            # Load original image
            original = cv2.imread(input_image)
            
            # Create preview with overlay
            preview = original.copy()
            
            # Add text overlay with conversion info
            cv2.putText(preview, "AI-Enhanced DXF Conversion", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save preview
            cv2.imwrite(preview_path, preview)
            
        except Exception as e:
            print(f"Error generating preview: {e}")
