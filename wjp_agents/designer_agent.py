import os
import yaml
import requests
import base64
import numpy as np
import cv2
from .utils.io_helpers import ensure_dirs, append_log, timestamp


class DesignerAgent:
    """
    Generates images from user requirements using OpenAI DALL-E API.
    Creates real AI-generated images based on prompts.
    """

    def __init__(self):
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.output_dir = os.path.join(project_root, "output", "designer")
        ensure_dirs([self.output_dir])
        
        # Load API key
        self.api_key = self._load_api_key()
        
    def _load_api_key(self):
        """Load OpenAI API key from config file or environment."""
        # Try environment variable first
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key
            
        # Try config file
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "api_keys.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    api_key = config.get("openai", {}).get("api_key")
                    if api_key and api_key != "your-api-key-here":
                        return api_key
        except Exception as e:
            print(f"Warning: Could not load API key from config: {e}")
            
        return None

    def run(self, user_input, options: dict | None = None):
        """Generate image(s) from user input using OpenAI DALL-E.

        options:
            - variations: int (1-4) number of images to request
            - negative_hints: str, appended as Avoid: ... in prompt
        """
        options = options or {}
        variations = int(max(1, min(4, options.get("variations", 1))))
        negative_hints = (options.get("negative_hints") or "").strip()

        base_prompt = f"High-quality top view render of waterjet flooring design: {user_input}"
        if negative_hints:
            base_prompt += f". Avoid: {negative_hints}"

        # Filenames container (support multiple when available)
        image_paths: list[str] = []

        # Try to generate image with OpenAI API
        if self.api_key:
            try:
                image_paths = self._generate_with_openai_multi(base_prompt, variations)
                if image_paths:
                    append_log({
                        "agent": "DesignerAgent",
                        "prompt": base_prompt,
                        "image_paths": image_paths,
                        "method": "OpenAI",
                        "variations": variations,
                    })
                    print(f"[DesignerAgent] Generated {len(image_paths)} AI image(s) -> {self.output_dir}")
                    return {"prompt": base_prompt, "image_paths": image_paths, "method": "OpenAI"}
            except Exception as e:
                print(f"[DesignerAgent] OpenAI generation failed: {e}")
                print(f"[DesignerAgent] Falling back to test image generation")

        # Fallback to test image if OpenAI fails
        image_filename = f"design_{timestamp()}.png"
        image_path = os.path.join(self.output_dir, image_filename)
        img = self._create_test_image(user_input)
        cv2.imwrite(image_path, img)

        append_log({"agent": "DesignerAgent", "prompt": base_prompt, "image_path": image_path, "method": "Test"})
        print(f"[DesignerAgent] Generated test image -> {image_path}")
        return {"prompt": base_prompt, "image_paths": [image_path], "method": "Test"}

    def _generate_with_openai(self, prompt, output_path):
        """Generate a single image using OpenAI DALL-E API (legacy single)."""
        if not self.api_key:
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Enhance prompt for waterjet design
            enhanced_prompt = self._enhance_prompt_for_waterjet(prompt)
            
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
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Get image URL or base64 data
                image_data = result["data"][0]
                
                if "url" in image_data:
                    # Download image from URL
                    img_response = requests.get(image_data["url"], timeout=30)
                    if img_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(img_response.content)
                        return True
                        
                elif "b64_json" in image_data:
                    # Save base64 image
                    image_bytes = base64.b64decode(image_data["b64_json"])
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    return True
                    
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return False

    def _generate_with_openai_multi(self, prompt: str, n: int) -> list[str]:
        """Generate multiple images (up to 4) using OpenAI DALL-E API.

        Returns list of saved image paths.
        """
        if not self.api_key:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        enhanced_prompt = self._enhance_prompt_for_waterjet(prompt)
        data = {
            "model": "dall-e-3",
            "prompt": enhanced_prompt,
            "n": int(max(1, min(4, n))),
            "size": "1024x1024",
            "quality": "standard",
            "style": "natural"
        }

        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code != 200:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            return []

        result = response.json()
        outputs = result.get("data", []) or []
        saved: list[str] = []
        for idx, item in enumerate(outputs):
            filename = f"design_{timestamp()}_{idx+1}.png" if len(outputs) > 1 else f"design_{timestamp()}.png"
            path = os.path.join(self.output_dir, filename)
            if "url" in item:
                r = requests.get(item["url"], timeout=30)
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    saved.append(path)
            elif "b64_json" in item:
                image_bytes = base64.b64decode(item["b64_json"])
                with open(path, "wb") as f:
                    f.write(image_bytes)
                saved.append(path)

        return saved

    def _enhance_prompt_for_waterjet(self, prompt):
        """Enhance prompt with waterjet-specific considerations."""
        enhancements = [
            "waterjet cutting safe design",
            "clean geometric lines",
            "minimum 3mm spacing between elements",
            "no floating parts",
            "continuous contours",
            "top view perspective",
            "high contrast black and white design",
            "suitable for stone cutting"
        ]
        
        enhanced = f"{prompt}, {', '.join(enhancements)}"
        return enhanced

    def _create_test_image(self, user_input):
        """Create a test image with geometric patterns (fallback)."""
        # Create a white canvas
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add geometric patterns based on input
        if "medallion" in user_input.lower():
            # Create a medallion pattern
            cv2.circle(img, (200, 200), 150, (0, 0, 0), 3)  # Outer circle
            cv2.circle(img, (200, 200), 100, (0, 0, 0), 2)  # Inner circle
            cv2.circle(img, (200, 200), 50, (0, 0, 0), 2)   # Center circle
            
            # Add decorative elements
            for i in range(8):
                angle = i * 45
                x = int(200 + 120 * np.cos(np.radians(angle)))
                y = int(200 + 120 * np.sin(np.radians(angle)))
                cv2.circle(img, (x, y), 20, (0, 0, 0), 2)
        
        elif "geometric" in user_input.lower():
            # Create geometric patterns
            # Squares
            cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
            cv2.rectangle(img, (250, 50), (350, 150), (0, 0, 0), 2)
            cv2.rectangle(img, (50, 250), (150, 350), (0, 0, 0), 2)
            cv2.rectangle(img, (250, 250), (350, 350), (0, 0, 0), 2)
            
            # Center pattern
            cv2.rectangle(img, (175, 175), (225, 225), (0, 0, 0), 2)
        
        else:
            # Default pattern - simple shapes
            cv2.circle(img, (100, 100), 50, (0, 0, 0), 2)
            cv2.rectangle(img, (250, 50), (350, 150), (0, 0, 0), 2)
            cv2.ellipse(img, (200, 300), (60, 30), 0, 0, 360, (0, 0, 0), 2)
        
        return img

    def generate_design(self, prompt):
        """Generate design image from prompt (for guided interface)."""
        return self.run(prompt)


